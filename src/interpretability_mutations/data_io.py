"""Load fold checkpoints, saved splits, candidate dictionaries, and RNA graphs."""

import pickle
from pathlib import Path

import dgl
import numpy as np
import torch

from ..HGTConv import load_model
from ..decoy_evaluation.decoy_utils import get_decoys


NUCLEOTIDE_TO_TYPE = {"A": 0, "C": 1, "G": 2, "N": 3, "U": 4}


def resolve_fold_runs(run_dir, folds):
    run_dir = Path(run_dir).expanduser().resolve()
    splits = sorted(run_dir.glob("splits_*.p"))
    if len(splits) != 1:
        raise ValueError(f"Expected one splits_*.p file in {run_dir}, found {len(splits)}.")
    template_fold = int(splits[0].stem.rsplit("_", 1)[1])
    try:
        prefix, suffix = run_dir.name.rsplit(f"_{template_fold}_", 1)
    except ValueError as error:
        raise ValueError(f"Run folder must contain _{template_fold}_ in its name: {run_dir.name}") from error
    runs = [(fold, run_dir.parent / f"{prefix}_{fold}_{suffix}") for fold in folds]
    for fold, folder in runs:
        if not (folder / f"splits_{fold}.p").is_file():
            raise FileNotFoundError(f"Missing fold {fold} split in {folder}.")
    return runs


def load_fold(run_dir, fold, device):
    with (run_dir / "meta.p").open("rb") as handle:
        meta = pickle.load(handle)
    with (run_dir / f"splits_{fold}.p").open("rb") as handle:
        splits = pickle.load(handle)
    checkpoint_path = run_dir / f"{run_dir.name}.pth"
    if not checkpoint_path.is_file():
        checkpoints = sorted(run_dir.glob("*.pth"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected one checkpoint in {run_dir}, found {len(checkpoints)}.")
        checkpoint_path = checkpoints[0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model, _ = load_model(str(run_dir), checkpoint_path.stem, device=str(device))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, meta, splits


def load_candidates(annotated_dir, decoy_mode, decoy_path=None):
    if decoy_mode == "internal":
        return get_decoys(mode="pdb", annots_dir=str(annotated_dir))
    if decoy_path is None:
        name = annotated_dir.name
        if not name.startswith("pockets_nx_") or not name.endswith("_annot"):
            raise ValueError("Specify the decoy dictionary path for this annotated directory.")
        fingerprint_type = name.removeprefix("pockets_nx_").split("_", 1)[0]
        data_dir = annotated_dir.parent.parent
        decoy_path = data_dir / "decoys" / f"filter_{fingerprint_type}_decoy_dict.p"
        if not decoy_path.is_file():
            matches = sorted(path for folder in ("decoys", "decoy")
                             for path in (data_dir / folder).glob(f"*{fingerprint_type}*decoy_dict.p"))
            if len(matches) != 1:
                raise FileNotFoundError("Cannot uniquely locate the candidate dictionary; specify --H_decoy_dict_path or --S_decoy_dict_path.")
            decoy_path = matches[0]
    with Path(decoy_path).expanduser().resolve().open("rb") as handle:
        return pickle.load(handle)


def selected_graph_names(split, candidates, graph_names=None, start_index=0, max_graphs=None):
    names = [str(name) for name in np.asarray(split).reshape(-1).tolist()]
    rankable = [name for name in names if len(name.split(":")) > 2 and name.split(":")[2] in candidates]
    skipped = len(names) - len(rankable)
    if graph_names:
        missing = sorted(set(graph_names) - set(rankable))
        if missing:
            raise ValueError(f"Requested graphs are absent from this rankable split: {missing}")
        rankable = [name for name in rankable if name in set(graph_names)]
    rankable = rankable[start_index:]
    if max_graphs is not None:
        rankable = rankable[:max_graphs]
    if not rankable:
        raise ValueError("No graphs remain after candidate matching and graph selection.")
    return rankable, skipped


def load_graph_sample(annotated_dir, graph_name, edge_map, device):
    with (annotated_dir / graph_name).open("rb") as handle:
        graph, _, _, _ = pickle.load(handle)
    graph = graph.to_directed()
    for _, attrs in graph.nodes(data=True):
        nucleotide = str(attrs.get("nt", "N")).upper()
        attrs["one_hot"] = torch.tensor(NUCLEOTIDE_TO_TYPE.get(nucleotide, 3), dtype=torch.long)
    nucleotides = []
    for node in sorted(graph.nodes()):
        nucleotide = str(graph.nodes[node].get("nt", "N")).upper()
        nucleotides.append(nucleotide if nucleotide in NUCLEOTIDE_TO_TYPE else "N")
    for _, _, attrs in graph.edges(data=True):
        label = attrs.get("label")
        if label not in edge_map:
            raise ValueError(f"Unknown edge label {label!r} in {graph_name}.")
        attrs["one_hot"] = torch.tensor(edge_map[label], dtype=torch.long)
    graph = dgl.from_networkx(graph, node_attrs=["one_hot"], edge_attrs=["one_hot"]).to(device)
    return graph, nucleotides
