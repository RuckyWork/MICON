import json
import random

import numpy as np
import torch

from .model import MICON, build_neighbors


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path, **overrides):
    config = json.loads(open(path).read())
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


@torch.no_grad()
def compute_scores(model, dataset, split="test", batch_size=64):
    """Anomaly scores of every slot in a split, shape (T_split, N)."""
    model.eval()
    slots = dataset.splits[split]
    return torch.cat([
        model.score(dataset.windows(slots[i : i + batch_size])).cpu() for i in range(0, len(slots), batch_size)
    ]).numpy()


def build_model(config, dataset):
    neighbor_indices, neighbor_mask = build_neighbors(
        dataset.adjacency, dataset.distance, config.get("num_neighbors"), config.get("fill_by_distance", False)
    )
    return MICON(
        neighbor_indices,
        neighbor_mask,
        window_size=config["window_size"],
        in_dim=dataset.n_features,
        hidden_dim=config["hidden_dim"],
        gru_hidden=config["gru_hidden"],
        embed_dim=config["embed_dim"],
        latent_dim=config["latent_dim"],
    )
