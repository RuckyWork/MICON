"""Train MICON on one dataset; the model of the final epoch is saved and evaluated on the test split.

Example:
    python train.py --config configs/nyc.json --seed 101
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from micon import FlowDataset, build_model, compute_scores, evaluate, load_config, micon_loss, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON config in configs/")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--data_dir", help="Override data_dir of the config")
    parser.add_argument("--epochs", type=int, help="Override epochs of the config")
    parser.add_argument("--output", help="Output directory (default: runs/<dataset>/seed<seed>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, dataset, config, split, optimizer=None, generator=None):
    """One pass over a split; trains when an optimizer is given, otherwise only reports the loss."""
    model.train(optimizer is not None)
    slots = dataset.splits[split]
    if optimizer is not None:
        slots = slots[torch.randperm(len(slots), generator=generator).numpy()]
    weights = {k: config[k] for k in ("lambda_dp", "lambda_var", "lambda_cov")}
    totals, n_batches = {}, 0
    for i in range(0, len(slots), config["batch_size"]):
        with torch.set_grad_enabled(optimizer is not None):
            R_c, R_s = model(dataset.windows(slots[i : i + config["batch_size"]]))
            loss, parts = micon_loss(R_c, R_s, model.predictor, **weights)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


def main():
    args = parse_args()
    config = load_config(args.config, data_dir=args.data_dir, epochs=args.epochs)
    output = Path(args.output or f"runs/{Path(config['data_dir']).name}/seed{args.seed}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    dataset = FlowDataset(config["data_dir"], config["window_size"], device=args.device)
    model = build_model(config, dataset).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    print(f"{dataset.meta['name']}: {dataset.n_slots} slots x {dataset.n_regions} regions, "
          f"train/val/test slots {[len(s) for s in dataset.splits.values()]}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    history = []
    for epoch in range(1, config["epochs"] + 1):
        train = run_epoch(model, dataset, config, "train", optimizer, generator)
        val = run_epoch(model, dataset, config, "val")
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {epoch:3d} | train loss {train['loss']:.4f} (dp {train['L_dp']:.4f} var {train['L_var']:.4f} "
              f"cov {train['L_cov']:.4f}) | val loss {val['loss']:.4f}", flush=True)

    torch.save({"config": config, "seed": args.seed, "state_dict": model.state_dict()}, output / "model.pt")
    (output / "history.json").write_text(json.dumps(history, indent=1))

    scores = compute_scores(model, dataset)
    np.save(output / "scores.npy", scores.astype(np.float32))
    test_start = dataset.meta["val_end"]
    result = evaluate(scores, dataset.test_labels(), dataset.events, test_start)
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    print("test | " + "  ".join(f"{k} {v:.2f}" for k, v in result.items()))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
