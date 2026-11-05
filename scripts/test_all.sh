#!/usr/bin/env bash
# Evaluate the released checkpoints, which reproduce the MICON results of the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
for dataset in nyc chicago nyc_sim; do
  echo "== $dataset"
  python test.py --checkpoint checkpoints/"$dataset"/seed*.pt
done
