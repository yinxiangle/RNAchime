import os
import pickle
import random

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


EDGE_TYPES = [
    "B53", "CWW", "CHH", "CHS", "CHW", "CSS", "CSW",
    "THH", "THS", "THW", "TSW", "TSS", "TWW"
]


class V1(Dataset):
    def __init__(self, annotated_path, shuffle=False, seed=None):
        self.path = annotated_path
        self.all_graphs = sorted(name for name in os.listdir(annotated_path) if os.path.isfile(os.path.join(annotated_path, name)))
        if not self.all_graphs:
            raise ValueError(f"No annotated graphs found in {annotated_path}")
        if shuffle:
            np.random.default_rng(seed).shuffle(self.all_graphs)

       
        self.edge_map = {label: i for i, label in enumerate(EDGE_TYPES)}
        self.num_edge_types = len(self.edge_map)
        self.nuc_map = {label: i for i, label in enumerate(['A', 'C', 'G', 'N', 'U'])}
        print(f">>> found {len(self.all_graphs)} graphs with {self.num_edge_types} edge types in {annotated_path}")

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        name = self.all_graphs[idx]
        with open(os.path.join(self.path, name), 'rb') as handle:
            graph, _, _, fp = pickle.load(handle)
        graph = graph.to_directed()

        fallback_count = 0
        for node, attrs in graph.nodes(data=True):
            label = attrs.get('nt')
            if label not in self.nuc_map:
                fallback_count += 1
                label = 'N'
            attrs['one_hot'] = torch.tensor(self.nuc_map[label], dtype=torch.long)
        if fallback_count:
            print(f"Warning: mapped {fallback_count} unknown/missing nt labels to 'N' in {name}")

        for source, target, attrs in graph.edges(data=True):
            label = attrs.get('label')
            if label not in self.edge_map:
                raise ValueError(f"Unknown edge label {label!r} on edge {(source, target)} in {name}")
            attrs['one_hot'] = torch.tensor(self.edge_map[label], dtype=torch.long)

        g_dgl = dgl.from_networkx(graph, edge_attrs=['one_hot'], node_attrs=['one_hot'])
        g_dgl.title = name
        return g_dgl, fp, idx


def collate_graphs(samples):
    """Batch graphs, float32 fingerprints, and one-dimensional sample indices."""
    graphs, fps, indices = zip(*samples)
    return dgl.batch(graphs), torch.as_tensor(np.asarray(fps), dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Loader:
    def __init__(self, annotated_path, batch_size=10, num_workers=8, shuffle=False, seed=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed if seed is not None and seed > 0 else None
        self.dataset = V1(annotated_path, shuffle=shuffle, seed=self.seed)
        self.all_graphs = self.dataset.all_graphs
        self.num_edge_types = self.dataset.num_edge_types

    def make_loader(self, dataset, shuffle=False):
        generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          collate_fn=collate_graphs, worker_init_fn=seed_worker, generator=generator)

    def get_data(self, k_fold=0):
        indices = np.arange(len(self.dataset))
        if k_fold > 1:
            from sklearn.model_selection import KFold
            folds = KFold(n_splits=k_fold, shuffle=True, random_state=self.seed).split(indices)
        else:
            split = int(0.8 * len(indices))
            folds = [(indices[:split], indices[split:])]

        for train_indices, test_indices in folds:
            train_set = Subset(self.dataset, train_indices.tolist())
            test_set = Subset(self.dataset, test_indices.tolist())
            yield self.make_loader(train_set, shuffle=True), self.make_loader(test_set)


class InferenceLoader(Loader):
    def __init__(self, annotated_path, batch_size=16, num_workers=8, seed=0):
        super().__init__(annotated_path, batch_size=batch_size, num_workers=num_workers, seed=seed)

    def get_data(self):
        return self.make_loader(self.dataset)
