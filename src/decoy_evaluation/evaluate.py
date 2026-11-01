import argparse
import csv
import pickle
from pathlib import Path
import numpy as np
import torch

from ..HGTConv import load_model, predict
from ..HGT_loader import InferenceLoader
from . import decoy_utils


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ligand_id(graph_name):
    fields = graph_name.split(':')
    if len(fields) < 3:
        raise ValueError(f'Cannot extract ligand ID from graph filename: {graph_name}')
    return fields[2]


def model_runs(model_root, kfold, time):
    if kfold < 1:
        raise ValueError('kfold must be positive.')
    if not model_root.is_dir():
        raise FileNotFoundError(f'Model directory does not exist: {model_root}')
    if (model_root / 'meta.p').is_file():
        split_paths = sorted(model_root.glob('splits_*.p'))
        if kfold != 1 or len(split_paths) != 1:
            raise ValueError('A single fold run folder requires --kfold 1 and exactly one saved split.')
        candidates, fold_ids = [model_root], [int(split_paths[0].stem.split('_')[-1])]
    else:
        candidates = sorted(path for path in model_root.iterdir() if path.is_dir())
        fold_ids = range(kfold)
    runs = []
    for fold in fold_ids:
        matches = [path for path in candidates if (path / 'meta.p').is_file() and (path / f'splits_{fold}.p').is_file()
                   and (time is None or path.name.endswith(f'_{time}'))]
        if len(matches) != 1:
            raise ValueError(f'Expected one model for fold {fold} in {model_root}, found {len(matches)}; check --model_root and --time.')
        run = matches[0]
        checkpoint = run / f'{run.name}.pth'
        if not checkpoint.is_file():
            weights = sorted(run.glob('*.pth'))
            if len(weights) != 1:
                raise ValueError(f'Expected one checkpoint in {run}, found {len(weights)}.')
            checkpoint = weights[0]
        runs.append((fold, run, checkpoint.stem))
    return runs


def compute_rank(kfold, annot_graph, out_path, model_root, mode='both', decoy_dict_path=None, time=None, device='auto', batch_size=128, workers=0):

    annot_graph, out_path, model_root = (Path(path).expanduser().resolve() for path in (annot_graph, out_path, model_root))
    runs = model_runs(model_root, kfold, time)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if str(device) == 'auto' else torch.device(device)

    if device.type == 'cuda' and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'Requested CUDA device is unavailable: {device}')

    decoy_sets = {}
    if mode in {'internal', 'both'}:
        decoy_sets['internal'] = decoy_utils.get_decoys(mode='pdb', annots_dir=annot_graph)
    if mode in {'external', 'both'}:
        decoy_dict_path = Path(decoy_dict_path).expanduser().resolve()
        print(f'Loading external decoy fingerprints from {decoy_dict_path}')
        with decoy_dict_path.open('rb') as handle:
            decoy_sets['external'] = pickle.load(handle)

    rows = {name: [] for name in decoy_sets}
    for fold, model_dir, checkpoint_name in runs:
        with (model_dir / f'splits_{fold}.p').open('rb') as handle:
            split = pickle.load(handle)
        test_graphs = [Path(str(name)).name for name in split['test']]
        train_graphs = {Path(str(name)).name for name in split['train']}
        if not test_graphs or train_graphs.intersection(test_graphs):
            raise ValueError(f'Fold {fold} has an empty test set or overlapping training and test graphs.')
        missing = [name for name in test_graphs if not (annot_graph / name).is_file()]
        if missing:
            raise FileNotFoundError(f'Test graphs missing from {annot_graph}: {missing[:5]}')

        # 每折只加载模型、预测一次，两种诱饵评估共享预测结果。
        print(f'>>> Processing fold {fold}: {model_dir}')
        model, meta = load_model(str(model_dir), checkpoint_name, device=device)
        inference = InferenceLoader(annot_graph, batch_size=batch_size, num_workers=workers)
        inference.dataset.all_graphs = test_graphs
        inference.all_graphs = test_graphs
        inference.dataset.edge_map = meta['edge_map']
        inference.dataset.num_edge_types = len(meta['edge_map'])
        inference.dataset.nuc_map = meta.get('nuc_map', inference.dataset.nuc_map)
        predictions, _, indices, graph_names = predict(model, inference.get_data(), max_graphs=len(test_graphs), device=device)
    
        for evaluation, decoy_dict in decoy_sets.items():
            evaluated, skipped = 0, 0
            for prediction, index in zip(predictions, indices):
                graph_name = graph_names[int(index)]
                ligand = ligand_id(graph_name)
                if ligand not in decoy_dict:
                    skipped += 1
                    continue
                true_fp, decoys = decoy_dict[ligand]
                rank = decoy_utils.euclidean_decoy_test_(fp_pred=prediction, true_fp=true_fp, decoys=decoys)
                rows[evaluation].append((graph_name, ligand, rank, fold))
                evaluated += 1
            if not evaluated:
                raise ValueError(f'No test graphs in fold {fold} have matching {evaluation} fingerprints.')
            print(f'Fold {fold}, {evaluation}: evaluated {evaluated}, skipped {skipped} without decoys.')


    out_path.mkdir(parents=True, exist_ok=True)
    results = {}
    for evaluation, evaluation_rows in rows.items():
        output_path = out_path / f'{model_root.name}_{time or "ranks"}_{evaluation}.txt'
        with output_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['graph', 'ligand', 'rank', 'fold'])
            writer.writerows(evaluation_rows)
        ranks = np.asarray([row[2] for row in evaluation_rows])
        mean_rank, std_rank = float(ranks.mean()), float(ranks.std())
        results[evaluation] = {'mean_rank': mean_rank, 'std_rank': std_rank, 'count': len(evaluation_rows), 'output_path': str(output_path)}
        print(f'{evaluation}: mean rank={mean_rank:.6f}, std={std_rank:.6f}\nRanking results saved to {output_path}')
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate annotated-ligand internal, external, or both.')
    parser.add_argument('--annot_graph', required=True, help='Annotated graph directory; relative paths use the current working directory.')
    parser.add_argument('--model_root', required=True, help='Experiment directory containing fold run folders, or one fold run folder.')
    parser.add_argument('--out_path', default=str(PROJECT_ROOT / 'outputs' / 'S/unimol_all'), help='Directory for per-mode result files.')
    parser.add_argument('--mode', choices=['internal', 'external', 'both'], default='both', help='Evaluation mode (default: both).')
    parser.add_argument('--decoy_dict_path', help='Decoy dictionary pickle, required for external and both modes.')
    parser.add_argument('--time', help='Run timestamp suffix; required when multiple matching runs exist.')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--device', default='cuda:0', help='auto, cpu, or cuda:0; respects CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode in {'external', 'both'} and not args.decoy_dict_path:
        parser.error('--decoy_dict_path is required when --mode is external or both.')
    return args


def main(argv=None):
    compute_rank(**vars(parse_args(argv)))


if __name__ == '__main__':
    main()
