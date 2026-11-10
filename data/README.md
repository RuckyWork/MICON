# Data format

Each dataset is a directory with two files.

`dataset.npz`:

| Key | Shape | Content |
|---|---|---|
| `flow` | (T, N, F) | Raw (unnormalized) traffic per slot and region; F = 2 (taxi outflow and inflow) for NYC and NYC-Simulated, F = 4 (taxi and bike-share outflow and inflow) for Chicago |
| `adjacency` | (N, N) | 0/1 region adjacency; the diagonal is ignored |
| `distance` | (N, N) | Distance between regions; only used when `num_neighbors` is an integer, to order the adjacent regions and, with `fill_by_distance`, to add the nearest non-adjacent ones |
| `labels` | (T, N) | 1 for an anomalous (slot, region) cell; only the test slots are used, for evaluation only |

`meta.json`:

| Key | Content |
|---|---|
| `train_end`, `val_end` | Train slots are `[W-1, train_end)`, validation `[train_end, val_end)`, test `[val_end, T)` |
| `start_time`, `freq_minutes` | Time of slot 0 and the slot length |
| `features` | Names of the F features |
| `events` (optional) | Test events `{"start", "end", "regions"}` (inclusive slots on the full time axis, region indices); only used for Hit@p |

The flow is normalized inside the code by `log1p` followed by a per-feature z-score, which is fitted on the training slots. Slot t is represented by the window of the W slots ending at t.

To use your own data, write these two files, copy one of the configs in `configs/`, set `data_dir`, and run `train.py`.
