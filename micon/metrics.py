"""Evaluation over all (test slot, region) cells: AUC and R@p rank them globally, Within_AUC per region."""

from math import ceil

import numpy as np
from sklearn.metrics import roc_auc_score

RECALL_BUDGETS = (1, 8, 20)
HIT_BUDGETS = (1, 3, 5, 8, 12, 20)


def evaluate(scores, labels, events=None, test_start=0):
    """Metrics in percent.

    scores, labels: (T_test, N). R@p is the share of labeled cells among the top p% of all cells.
    Within_AUC is the ROC-AUC computed separately for each region that has both labeled and
    unlabeled test cells, averaged over these regions.
    Hit@p (only when events are given) is the share of events with at least one of their cells in the top p%.
    Event start/end are slots on the full time axis; test_start converts them to test rows.
    """
    flat, y = scores.ravel(), labels.ravel().astype(bool)
    order = np.argsort(-flat, kind="stable")
    result = {"AUC": 100 * roc_auc_score(y, flat)}
    for p in RECALL_BUDGETS:
        result[f"R@{p}%"] = 100 * y[order[: ceil(flat.size * p / 100)]].sum() / y.sum()
    labels = labels.astype(bool)
    regions = [r for r in range(labels.shape[1]) if 0 < labels[:, r].sum() < len(labels)]
    result["Within_AUC"] = 100 * np.mean([roc_auc_score(labels[:, r], scores[:, r]) for r in regions])
    if events:
        for p in HIT_BUDGETS:
            selected = np.zeros(flat.size, dtype=bool)
            selected[order[: int(flat.size * p / 100)]] = True
            selected = selected.reshape(scores.shape)
            hits = sum(
                selected[e["start"] - test_start : e["end"] - test_start + 1, e["regions"]].any() for e in events
            )
            result[f"Hit@{p}%"] = 100 * hits / len(events)
    return result
