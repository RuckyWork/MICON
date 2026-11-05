"""Score the test split with trained checkpoints and report AUC, R@p and (if events are given) Hit@p.

With several checkpoints (e.g. one per seed) the mean and standard deviation are printed as well.

Example:
    python test.py --checkpoint checkpoints/nyc/seed*.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from micon import FlowDataset, build_model, compute_scores, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--data_dir", help="Override the data_dir stored in the checkpoint")
    parser.add_argument("--save_scores", action="store_true", help="Save <checkpoint>.scores.npy, shape (T_test, N)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets, rows = {}, []
    for path in args.checkpoint:
        ckpt = torch.load(path, map_location=args.device)
        config = ckpt["config"]
        key = (args.data_dir or config["data_dir"], config["window_size"])
        if key not in datasets:
            datasets[key] = FlowDataset(*key, device=args.device)
        dataset = datasets[key]
        model = build_model(config, dataset).to(args.device)
        model.load_state_dict(ckpt["state_dict"])

        scores = compute_scores(model, dataset)
        if args.save_scores:
            np.save(Path(path).with_suffix(".scores.npy"), scores.astype(np.float32))
        result = evaluate(scores, dataset.test_labels(), dataset.events, dataset.meta["val_end"])
        rows.append(result)
        print(f"{path} | " + "  ".join(f"{k} {v:.2f}" for k, v in result.items()))

    if len(rows) > 1:
        summary = {k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows], ddof=1)) for k in rows[0]}
        print(f"mean ± std over {len(rows)} | " + "  ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))
        print(json.dumps({k: round(m, 2) for k, (m, _) in summary.items()}))


if __name__ == "__main__":
    main()
