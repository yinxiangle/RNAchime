"""Run nucleotide mutations """

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.interpretability_mutations"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data_io import load_candidates, load_fold, load_graph_sample, resolve_fold_runs, selected_graph_names
from .mutation_utils import mutation_effects


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run nucleotide mutations and write final per-graph CSV tables only.")
    parser.add_argument("--datasets", nargs="+", choices=["H", "S"], default=["H", "S"])
    for dataset in ("H", "S"):
        parser.add_argument(f"--{dataset}_run_dir", type=Path, help="One fold folder used to locate the requested folds.")
        parser.add_argument(f"--{dataset}_annotated_data", type=Path, help="Override the annotated directory in meta.p.")
        parser.add_argument(f"--{dataset}_decoy_dict_path", type=Path, help="External decoy dictionary; inferred from the annotated directory if omitted.")
    parser.add_argument("--output_dir", type=Path, help="Output directory; defaults to outputs_<decoy_mode>_<node_selection> beside this script.")
    parser.add_argument("--node_counts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--partition", choices=["train", "test"], default="test")
    parser.add_argument("--graph_name", action="append", help="Restrict a fold to the named graph; repeat to select several.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_graphs", type=int, help="Maximum selected graphs per fold for a short check.")
    parser.add_argument("--mutation_mode", choices=["joint", "independent"], default="joint")
    parser.add_argument("--node_selection", choices=["random", "degree"], default="random")
    parser.add_argument("--decoy_mode", choices=["external", "internal"], default="external")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:N.")
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        if getattr(args, f"{dataset}_run_dir") is None:
            parser.error(f"--{dataset}_run_dir is required when selecting {dataset}.")
    if not args.node_counts or min(args.node_counts) < 1 or len(set(args.node_counts)) != len(args.node_counts):
        parser.error("node_counts must contain distinct positive integers.")
    if min(args.folds) < 0 or len(set(args.folds)) != len(args.folds):
        parser.error("folds must contain distinct nonnegative integers.")
    if args.start_index < 0 or (args.max_graphs is not None and args.max_graphs < 1):
        parser.error("start_index must be nonnegative and max_graphs must be positive.")
    return args


def run_dataset(args, dataset, device):
    runs = resolve_fold_runs(getattr(args, f"{dataset}_run_dir"), args.folds)
    annotated_override = getattr(args, f"{dataset}_annotated_data")
    decoy_path = getattr(args, f"{dataset}_decoy_dict_path")
    rows, candidate_cache = [], {}
    for fold, folder in runs:
        print(f"{dataset}, fold {fold}: {folder}", flush=True)
        model, meta, splits = load_fold(folder, fold, device)
        annotated_dir = Path(annotated_override or meta["annotated_data"]).expanduser().resolve()
        if not annotated_dir.is_dir():
            raise FileNotFoundError(f"Annotated directory does not exist: {annotated_dir}; set --{dataset}_annotated_data.")
        if annotated_dir not in candidate_cache:
            candidate_cache[annotated_dir] = load_candidates(annotated_dir, args.decoy_mode, decoy_path)
        candidates = candidate_cache[annotated_dir]
        graph_names, missing_decoys = selected_graph_names(
            splits[args.partition], candidates, args.graph_name, args.start_index, args.max_graphs,
        )
        skipped_small = 0
        for graph_name in tqdm(graph_names, desc=f"{dataset} fold {fold}"):
            graph, nucleotides = load_graph_sample(annotated_dir, graph_name, meta["edge_map"], device)
            effects = mutation_effects(
                model, graph, nucleotides, graph_name, candidates, args.node_counts,
                args.mutation_mode, args.node_selection, args.seed,
            )
            if effects is None:
                skipped_small += 1
                continue
            rows.append({"graph_name": graph_name, **effects})
        print(f"{dataset}, fold {fold}: skipped {missing_decoys} without decoys and {skipped_small} without enough standard nodes.")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not rows:
        raise ValueError(f"No graphs were evaluated for {dataset}.")
    table = pd.DataFrame(rows).sort_values("graph_name", kind="stable").reset_index(drop=True)
    if table["graph_name"].duplicated().any():
        raise ValueError(f"Selected {dataset} splits contain duplicate graphs; final CSV requires one row per graph.")
    return table


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"Requested CUDA device is unavailable: {device}.")
    output_dir = args.output_dir or Path(__file__).resolve().parent / f"outputs_{args.decoy_mode}_{args.node_selection}"
    output_dir = output_dir.expanduser().resolve()
    tables = {dataset: run_dataset(args, dataset, device) for dataset in args.datasets}
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, table in tables.items():
        output = output_dir / f"{dataset}_{args.decoy_mode}_{args.node_selection}.csv"
        table.to_csv(output, index=False)
        print(f"Wrote {len(table)} graphs to {output}")


if __name__ == "__main__":
    main()
