import argparse
from datetime import datetime
import os
import pickle
import random

if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'RNAchime.src'

import dgl
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import learn
from .HGT_loader import Loader
from .HGTConv import Model
from .utils import mkdirs, save_hyperparameter_summary, get_external_split_data


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train RNAchime on annotated RNA graphs.')
    parser.add_argument('-da', '--annotated_data', required=True, help='Directory containing annotated graph pickles.')
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-nw', '--workers', type=int, default=10)
    parser.add_argument('-dir', '--save_dir', default='S_5fold', help='Run group under RNAchime/results, or an absolute output directory.')
    parser.add_argument('-n', '--name', default='unimol_all', help='Experiment name.')
    parser.add_argument('-ep', '--num_epochs', type=int, default=200)
    parser.add_argument('-ad', '--attributor_dims', nargs='+', type=int, default=[32, 512])
    parser.add_argument('-hs', '--head_size', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('-nh', '--num_heads', nargs='+', type=int, default=[1, 1, 1])
    parser.add_argument('-w', '--warm_start', help='Checkpoint used to initialize compatible embedding weights.')
    parser.add_argument('-de', '--delta', type=float, default=1.0)
    parser.add_argument('-do', '--dropout', type=float, default=0.2)
    parser.add_argument('-po', '--pool', choices=['max', 'sum', 'att'], default='max')
    parser.add_argument('-rs', '--seed', type=int, default=11, help='Positive seed for reproducibility; 0 disables fixed seeding.')
    parser.add_argument('-kf', '--kfold', type=int, default=5, help='Number of folds; 0 or 1 uses an 80/20 split.')
    parser.add_argument('-es', '--early_stop', type=int, default=15)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00001)
    parser.add_argument('-cr', '--criterion', default='HuberLoss', help='BCELoss for fingerprints, HuberLoss for embeddings.')
    parser.add_argument('--device', default='cuda:0', help='auto, cpu, or a CUDA device such as cuda:0.')
    parser.add_argument('--split_dir', help='External split directory or one split pickle file.')
    parser.add_argument('--run_label', help='Optional prefix for each run folder.')
    args = parser.parse_args(argv)

    if len(args.head_size) != len(args.num_heads) or len(args.attributor_dims) < 2:
        parser.error('head_size and num_heads must have equal lengths; attributor_dims needs at least two values.')
    if args.head_size[-1] * args.num_heads[-1] != args.attributor_dims[0]:
        parser.error('The final HGT embedding width must match the first attributor dimension.')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.seed > 0:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'Requested CUDA device is unavailable: {device}')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    used_gpus_count = int(device.type == 'cuda')
    print('OPTIONS USED')
    print('\n'.join(f'{key}: {value}' for key, value in vars(args).items()))
    print(f'Using device {device}')

    if args.seed > 0:
        torch.set_num_threads(1)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        dgl.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    criterion_name = args.criterion.strip().lower()
    if criterion_name in {'huberloss', 'huber'}:
        criterion = torch.nn.HuberLoss(delta=args.delta)
    elif criterion_name in {'bceloss', 'bce'}:
        criterion = torch.nn.BCELoss()
    else:
        raise ValueError(f'Unsupported criterion: {args.criterion}')
    args.criterion = criterion.__class__.__name__

    loader = Loader(args.annotated_data, batch_size=args.batch_size, num_workers=args.workers, seed=args.seed)
   
    fingerprint = np.asarray(loader.dataset[0][1])
    if fingerprint.ndim != 1 or fingerprint.shape[0] != args.attributor_dims[-1]:
        raise ValueError(f'Fingerprint shape {fingerprint.shape} does not match output width {args.attributor_dims[-1]}.')

    data = get_external_split_data(loader, args.split_dir, k_folds=args.kfold) if args.split_dir else loader.get_data(k_fold=args.kfold)
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    for fold, (train_loader, test_loader) in enumerate(data):
        train_indices, test_indices = train_loader.dataset.indices, test_loader.dataset.indices
        if not len(train_indices) or not len(test_indices):
            raise ValueError(f'Fold {fold} has an empty training or test set.')
        if set(train_indices) & set(test_indices):
            raise ValueError(f'Fold {fold} has overlapping training and test graphs.')
        model = Model(args.head_size, args.num_heads, device, criterion, args.attributor_dims,
                      num_ntypes=len(loader.dataset.nuc_map), num_etypes=loader.num_edge_types,
                      pool=args.pool, dropout=args.dropout)
        if args.warm_start:
            state = torch.load(args.warm_start, map_location='cpu')['model_state_dict']
            target = model.state_dict()
            compatible = {key: value for key, value in state.items() if key.startswith('embedder.') and key in target and value.shape == target[key].shape}
            if not compatible:
                raise ValueError('The warm-start checkpoint has no compatible embedding weights.')
            model.load_state_dict(compatible, strict=False)
            print(f'Loaded {len(compatible)} compatible embedding tensors from {args.warm_start}')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        name = f'{args.run_label or args.name}_{fold}_{run_time}'
        result_folder, save_path = mkdirs(args.save_dir, args.name, name, permissive=False)
        print(f'Fold {fold}: train={len(train_indices)}, test={len(test_indices)}, saving to {result_folder}')

        meta = vars(args).copy()
        meta.update(edge_map=loader.dataset.edge_map, nuc_map=loader.dataset.nuc_map, device=str(device))
        with open(os.path.join(result_folder, 'meta.p'), 'wb') as handle:
            pickle.dump(meta, handle)
        all_graphs = np.asarray(loader.all_graphs)
        with open(os.path.join(result_folder, f'splits_{fold}.p'), 'wb') as handle:
            pickle.dump({'train': all_graphs[train_indices], 'test': all_graphs[test_indices]}, handle)
        save_hyperparameter_summary(args, loader, model, device, used_gpus_count, result_folder, fold, run_time)

        writer = SummaryWriter(result_folder)
        try:
            learn.train_model(model, optimizer, device, train_loader, test_loader, save_path, writer=writer,
                              num_epochs=args.num_epochs, early_stop_threshold=args.early_stop, run_time=run_time)
        finally:
            writer.close()


if __name__ == '__main__':
    main()
