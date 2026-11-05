"""MICON model: self view, context view, projection heads and the dual predictor."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_neighbors(adjacency, distance, k, fill_by_distance=False):
    """Select up to k geographic neighbors per region for the context view.

    Adjacent regions are taken first, ordered by distance. If a region has fewer than k
    and fill_by_distance is set, the nearest non-adjacent regions fill the remaining slots.
    Unused slots point to the region itself and are masked out.

    Returns neighbor indices (N, k) int64 and a validity mask (N, k) bool.
    """
    adjacency = np.asarray(adjacency)
    distance = np.asarray(distance, dtype=np.float64)
    n = adjacency.shape[0]
    if adjacency.shape != (n, n) or distance.shape != (n, n):
        raise ValueError(f"adjacency and distance must be ({n}, {n})")
    indices = np.tile(np.arange(n)[:, None], (1, k))
    mask = np.zeros((n, k), dtype=bool)
    for i in range(n):
        others = np.arange(n) != i
        adjacent = np.flatnonzero((adjacency[i] > 0) & others)
        neighbors = adjacent[np.argsort(distance[i, adjacent], kind="stable")][:k]
        if fill_by_distance and len(neighbors) < k:
            rest = np.flatnonzero((adjacency[i] <= 0) & others)
            rest = rest[np.argsort(distance[i, rest], kind="stable")]
            neighbors = np.concatenate([neighbors, rest[: k - len(neighbors)]])
        indices[i, : len(neighbors)] = neighbors
        mask[i, : len(neighbors)] = True
    return indices.astype(np.int64), mask


class FeatureBatchNorm(nn.Module):
    """BatchNorm over the last dimension, with statistics shared across all leading dimensions."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        return self.bn(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class ContextEncoder(nn.Module):
    """Context view: infers the state of region i from its neighbors only, never from region i itself.

    alpha_ij = softmax_j(q_i . k_j / sqrt(h)),  Z_i = sum_j alpha_ij v_j
    The query q_i is a learnable per-region vector, so no observation of region i enters this view.
    """

    def __init__(self, neighbor_indices, neighbor_mask, window_size, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(neighbor_indices, dtype=torch.long))
        self.register_buffer("mask", torch.as_tensor(neighbor_mask, dtype=torch.bool))
        self.hidden_dim = hidden_dim
        self.W_k = nn.Linear(window_size * in_dim, hidden_dim)
        self.W_v = nn.Linear(window_size * in_dim, out_dim)
        self.region_query = nn.Parameter(torch.randn(len(self.indices), hidden_dim) * 0.02)
        self.output_norm = FeatureBatchNorm(out_dim)

    def forward(self, x):
        """x: (B, N, W, F) -> (B, N, out_dim)."""
        flat = x.flatten(-2)
        keys = self.W_k(flat)[:, self.indices]  # (B, N, k, hidden)
        values = self.W_v(flat)[:, self.indices]  # (B, N, k, out)
        logits = (self.region_query.unsqueeze(1) * keys).sum(-1) / math.sqrt(self.hidden_dim)
        weights = torch.softmax(logits.masked_fill(~self.mask, float("-inf")), dim=-1)
        # A region without any neighbor gets an all -inf row, whose softmax is NaN.
        weights = torch.nan_to_num(weights, 0.0)
        return self.output_norm((weights.unsqueeze(-1) * values).sum(2))


class SelfEncoder(nn.Module):
    """Self view: a GRU over each region's own window, read out at the last step."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.output_norm = FeatureBatchNorm(out_dim)

    def forward(self, x):
        """x: (B, N, W, F) -> (B, N, out_dim)."""
        b, n, w, f = x.shape
        # cuDNN's GRU needs far more memory for B * N short sequences than the native kernel.
        with torch.backends.cudnn.flags(enabled=False):
            out, _ = self.gru(x.reshape(b * n, w, f))
        return self.output_norm(self.head(out[:, -1].reshape(b, n, -1)))


class ProjectionHead(nn.Module):
    """Two-layer MLP that maps a view representation into the shared C-dimensional latent space."""

    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def _predictor_mlp(dim):
    return nn.Sequential(
        nn.Linear(dim, dim), FeatureBatchNorm(dim), nn.ReLU(),
        nn.Linear(dim, dim), FeatureBatchNorm(dim), nn.ReLU(),
        nn.Linear(dim, dim),
    )


class DualPredictor(nn.Module):
    """F^(c) predicts R^s from R^c and F^(s) predicts R^c from R^s."""

    def __init__(self, dim):
        super().__init__()
        self.f_context = _predictor_mlp(dim)
        self.f_self = _predictor_mlp(dim)

    def forward(self, R_c, R_s):
        """Per-instance dual prediction error, (..., C) -> (...).

        Targets are not detached, so the error trains both views.
        """
        err_s = F.mse_loss(self.f_context(R_c), R_s, reduction="none").mean(-1)
        err_c = F.mse_loss(self.f_self(R_s), R_c, reduction="none").mean(-1)
        return err_s + err_c


class MICON(nn.Module):
    def __init__(
        self,
        neighbor_indices,
        neighbor_mask,
        window_size=12,
        in_dim=2,
        hidden_dim=32,
        gru_hidden=64,
        embed_dim=128,
        latent_dim=128,
    ):
        super().__init__()
        self.context_encoder = ContextEncoder(
            neighbor_indices, neighbor_mask, window_size, in_dim, hidden_dim, embed_dim
        )
        self.self_encoder = SelfEncoder(in_dim, gru_hidden, embed_dim)
        self.context_head = ProjectionHead(embed_dim, latent_dim)
        self.self_head = ProjectionHead(embed_dim, latent_dim)
        self.predictor = DualPredictor(latent_dim)

    def forward(self, x):
        """x: (B, N, W, F) windows ending at the scored slot -> latent representations R^c, R^s of shape (B, N, C)."""
        return self.context_head(self.context_encoder(x)), self.self_head(self.self_encoder(x))

    @torch.no_grad()
    def score(self, x):
        """Anomaly score S(i, t): the dual prediction error, (B, N, W, F) -> (B, N)."""
        return self.predictor(*self(x))
