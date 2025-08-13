import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch


def make_random_transition_matrix(n_nodes, alpha):
    return np.random.dirichlet([alpha] * n_nodes, size=n_nodes)


def random_walk(matrix, length):
    """
    Generate a random walk of a given length.
    Trick: Pre-generate many random numbers to speed up the process.
    """
    cprobs = matrix.cumsum(axis=1)  # Prepare for searchsorted
    rand = np.random.rand(length)  # Pre-generate many random numbers
    N = len(matrix)

    walk = np.zeros(length, dtype=int)
    walk[0] = random.randint(0, N - 1)  # Random starting point
    for i in range(1, length):
        walk[i] = np.searchsorted(cprobs[walk[i - 1]], rand[i])
    return np.array(walk)


def get_perm_and_reverse(n_nodes):
    perm = np.random.permutation(n_nodes)
    rev_perm = np.empty_like(perm)
    rev_perm[perm] = np.arange(len(perm))
    return perm, rev_perm


def get_probs(walk, xi, k, alpha):
    counts = np.zeros((k,))

    # xi is the last node in the walk
    positions = np.where(walk == xi)[0][:-1]
    positions += 1  # Look at the next token

    values = walk[positions]
    np.add.at(counts, values, 1)

    return (counts + alpha) / (counts.sum() + alpha * k)


class Dataset:
    def __init__(self, data_cfg, matrices, length):
        # Parameters of the random walk
        self.n_nodes = data_cfg.n_nodes
        self.alpha = data_cfg.alpha

        self.walk_len = data_cfg.walk_len

        self.use_remap = data_cfg.use_remap
        self.metrics_cut = data_cfg.metrics_cut

        print(
            f"Dataset: {length} samples, {self.n_nodes} nodes, alpha={self.alpha}, walk_len={self.walk_len}, use_remap={self.use_remap}"
        )
        print("Predefined matrices:", len(matrices) if matrices is not None else 0)
        self.matrices = matrices
        self.n_matrices = len(matrices) if matrices is not None else -1

        self._len = length

    def generate_walk(self, idx):
        if self.n_matrices == -1:
            # Just generate a matrix online.
            mat = make_random_transition_matrix(n_nodes=self.n_nodes, alpha=self.alpha)
        else:
            # Or take one from the predefined matrices.
            mat = self.matrices[idx % len(self.matrices)]
        walk = random_walk(mat, self.walk_len)

        return mat, walk

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        pmatrix, walk = self.generate_walk(idx)

        if self.use_remap:
            remap, rev_remap = get_perm_and_reverse(self.n_nodes)
            walk = remap[walk]
            pmatrix = pmatrix[rev_remap][:, rev_remap]

        target = walk[1:].copy()  # Next node
        walk = walk[:-1]  # Drop last item for training

        # True distribution of next nodes for the last sequence
        cut_target = target[-self.metrics_cut :]
        oracle_dist = pmatrix[walk[-self.metrics_cut :]]

        # Empirical estimation of the distribution
        empirical_dist = np.stack(
            [
                get_probs(walk[: i + 1], walk[i], self.n_nodes, self.alpha)
                for i in range(-self.metrics_cut, -1)
            ]
            + [get_probs(walk, walk[-1], self.n_nodes, self.alpha)],
            axis=0,
        )

        return {
            "walk": walk,
            "target": target,
            "cut_target": cut_target,
            "oracle": oracle_dist,
            "empirical": empirical_dist,
        }


def make_datasets(cfg):
    ds_len = cfg.data.batch_size * cfg.train.total_steps

    # Training dataset
    N = cfg.data.training_matrices
    if N == -1:
        training_matrices = None
    else:
        training_matrices = [
            make_random_transition_matrix(
                n_nodes=cfg.data.n_nodes, alpha=cfg.data.alpha
            )
            for _ in range(N)
        ]

    training_dataset = Dataset(cfg.data, training_matrices, ds_len)

    # Validation dataset
    valid_matrices = [
        make_random_transition_matrix(n_nodes=cfg.data.n_nodes, alpha=cfg.data.alpha)
        for _ in range(cfg.data.validation_matrices)
    ]
    valid_dataset = Dataset(cfg.data, valid_matrices, cfg.data.validation_matrices)

    train_dataloader = DataLoader(
        training_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )

    return train_dataloader, valid_dataloader
