"""Select nucleotide nodes and measure rank changes after type substitutions."""

import itertools
import random

import numpy as np
import torch

from ..decoy_evaluation.decoy_utils import euclidean_decoy_test_
from .data_io import NUCLEOTIDE_TO_TYPE


STANDARD_NUCLEOTIDES = ("A", "C", "G", "U")
MIN_JOINT_STANDARD_NODES = 10


def select_mutation_nodes(nucleotides, degrees, graph_name, nodes_per_graph, selection_method, seed):
    eligible = [index for index, nucleotide in enumerate(nucleotides) if nucleotide in STANDARD_NUCLEOTIDES]
    if nodes_per_graph is None or nodes_per_graph >= len(eligible):
        return eligible
    if selection_method == "random":
        # Match the original per-graph sampling exactly for every node count.
        rng = random.Random(f"{seed}:{graph_name}")
        return sorted(rng.sample(eligible, nodes_per_graph))
    if selection_method == "degree":
        return sorted(eligible, key=lambda index: (-int(degrees[index]), index))[:nodes_per_graph]
    raise ValueError(f"Unsupported node selection method: {selection_method}")


def iter_mutation_combinations(selected_nodes, nucleotides, mutation_mode):
    if mutation_mode == "independent":
        for node in selected_nodes:
            for nucleotide in STANDARD_NUCLEOTIDES:
                if nucleotide != nucleotides[node]:
                    yield (node,), (nucleotide,)
    elif mutation_mode == "joint":
        alternatives = [tuple(nt for nt in STANDARD_NUCLEOTIDES if nt != nucleotides[node]) for node in selected_nodes]
        for replacements in itertools.product(*alternatives):
            yield tuple(selected_nodes), replacements
    else:
        raise ValueError(f"Unsupported mutation mode: {mutation_mode}")


def prediction_rank(model, graph, true_fp, decoys):
    prediction, _ = model(graph)
    rank = euclidean_decoy_test_(prediction.detach().cpu().numpy().reshape(-1), true_fp, decoys)
    return rank


@torch.no_grad()
def mutation_effects(model, graph, nucleotides, graph_name, candidates, node_counts, mutation_mode, node_selection, seed):
    eligible_count = sum(nt in STANDARD_NUCLEOTIDES for nt in nucleotides)
    if eligible_count == 0 or (mutation_mode == "joint" and eligible_count < MIN_JOINT_STANDARD_NODES):
        return None
    true_fp, decoys = candidates[graph_name.split(":")[2]]
    if not len(decoys):
        raise ValueError(f"Empty candidate pool for {graph_name}.")
    baseline_rank = prediction_rank(model, graph, true_fp, decoys)
    degrees = graph.in_degrees().detach().cpu()
    effects = {}
    for count in node_counts:
        selected = select_mutation_nodes(nucleotides, degrees, graph_name, count, node_selection, seed)
        deltas = []
        for nodes, replacements in iter_mutation_combinations(selected, nucleotides, mutation_mode):
            # Every substitution starts from the original graph; topology stays fixed.
            with graph.local_scope():
                node_types = graph.ndata["one_hot"].clone()
                for node, nucleotide in zip(nodes, replacements):
                    node_types[node] = NUCLEOTIDE_TO_TYPE[nucleotide]
                graph.ndata["one_hot"] = node_types
                rank = prediction_rank(model, graph, true_fp, decoys)
                deltas.append(abs(rank - baseline_rank))
        effects[f"nodes_per_graph_{count}"] = float(np.mean(deltas))
    return effects
