"""Dataset loading: one directory per dataset with dataset.npz and meta.json (format in data/README.md)."""

import json
from pathlib import Path

import numpy as np
import torch


def normalize(flow, train_end):
    """log1p followed by a per-feature z-score fitted on the training slots."""
    train = np.log1p(flow[:train_end].reshape(-1, flow.shape[-1]))
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.clip(train.std(axis=0, dtype=np.float64).astype(np.float32), 1e-6, None)
    return ((np.log1p(flow) - mean) / std).astype(np.float32)


class FlowDataset:
    """Normalized flow on the full time axis plus the graph and labels.

    Slot t is represented by the window of the W slots ending at t. Training uses slots
    [W-1, train_end), validation [train_end, val_end) and testing [val_end, T).
    """

    def __init__(self, data_dir, window_size=12, device="cpu"):
        data_dir = Path(data_dir)
        self.meta = json.loads((data_dir / "meta.json").read_text())
        arrays = np.load(data_dir / "dataset.npz")
        flow = arrays["flow"].astype(np.float32)
        self.adjacency = arrays["adjacency"]
        self.distance = arrays["distance"]
        self.labels = arrays["labels"].astype(bool)
        self.window_size = window_size
        self.n_slots, self.n_regions, self.n_features = flow.shape

        train_end, val_end = self.meta["train_end"], self.meta["val_end"]
        if not window_size <= train_end < val_end < self.n_slots:
            raise ValueError(f"invalid split: W={window_size}, train_end={train_end}, val_end={val_end}")
        self.flow = torch.from_numpy(normalize(flow, train_end)).to(device)
        self.splits = {
            "train": np.arange(window_size - 1, train_end),
            "val": np.arange(train_end, val_end),
            "test": np.arange(val_end, self.n_slots),
        }
        # Numeric test events (start/end slot on the full time axis, regions), only used for Hit@p.
        self.events = self.meta.get("events")

    def windows(self, slots):
        """Windows for the given target slots, shape (B, N, W, F)."""
        slots = torch.as_tensor(np.asarray(slots), device=self.flow.device)
        offsets = torch.arange(-self.window_size + 1, 1, device=self.flow.device)
        return self.flow[slots[:, None] + offsets].permute(0, 2, 1, 3)

    def test_labels(self):
        return self.labels[self.splits["test"]]
