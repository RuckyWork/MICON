#!/usr/bin/env bash
# Train MICON on every dataset with the three seeds of the paper; results go to runs/<dataset>/seed<seed>/.
set -euo pipefail
cd "$(dirname "$0")/.."
for dataset in nyc chicago nyc_sim; do
  for seed in 101 202 303; do
    python train.py --config "configs/$dataset.json" --seed "$seed"
  done
  python test.py --checkpoint runs/"$dataset"/seed*/model.pt
done
