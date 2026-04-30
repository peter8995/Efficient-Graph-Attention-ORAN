# Efficient Graph Attention-based Learning for Traffic Prediction and Uncertainty-Aware Anomaly Detection in AI-driven O-RAN

A multi-domain (temporal + spatial + spectral) deep-learning framework for per-slice traffic prediction and post-hoc anomaly detection on the Colosseum O-RAN dataset.

## Overview

The model fuses three complementary branches over a feature-as-node graph and produces (a) a one-step-ahead traffic prediction (`sum_granted_prbs`) and (b) Chebyshev-thresholded anomaly flags from prediction residuals.

| Branch       | Purpose                                                                                | Key components |
|--------------|----------------------------------------------------------------------------------------|----------------|
| BiLSTM       | Per-node shared temporal encoder over the sequence axis                                | `Linear(1,16) → BiLSTM(hidden=64, layers=2)` |
| GAT          | Spatial attention on a 16-node feature graph (binary fully-connected + self-loop)      | Custom `GATLayer` (no `torch_geometric`), multi-head, mean / concat merge |
| FFT-Transformer | Magnitude / phase dual transformer encoders on the rFFT spectrum (T=15→8 bins)       | Learnable positional embedding, `norm_first=True`, mean pooling |
| Fusion       | Concatenate branches → mean-pool over N nodes → scalar regression                      | `MLP` head |
| Anomaly      | Chebyshev k-σ threshold on signed / abs residuals (post-hoc, no feedback into model)   | k ∈ {2, 3} default |

## Repository layout

```
.
├── DataProcessor.py        # CSV pipeline, RobustScaler, ChunkShuffleSampler, adjacency building
├── model.py                # TrafficModel (BiLSTM + GAT + FFT-Transformer + fusion)
├── anomaly.py              # Chebyshev calibration + overlay plotting helpers
├── train.py                # Training, evaluation, checkpointing, anomaly export
├── scripts/
│   ├── clean_dataset.py    # Strip non-metrics files & flatten nested dirs
│   └── group_by_slice.py   # Reorganise CSVs into embb/mtc/urllc by IMSI UE ID
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone & create environment

```bash
git clone <YOUR_GIT_URL> Efficient-Graph-Attention-ORAN
cd Efficient-Graph-Attention-ORAN

# Conda
conda create -n egat-oran python=3.10 -y
conda activate egat-oran
pip install -r requirements.txt

# OR venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended (training auto-falls back to CPU via `--device cpu`).

### 2. Get the dataset

The training script expects the **Colosseum O-RAN ColORAN** dataset:

```
<DATA_ROOT>/colosseum-oran-coloran-dataset/tr{N}/exp*/bs*/{embb|mtc|urllc}/*metrics.csv
```

Download from the official source (https://github.com/wineslab/colosseum-oran-coloran-dataset) or symlink an existing copy:

```bash
mkdir -p Dataset
ln -s /absolute/path/to/colosseum-oran-coloran-dataset Dataset/colosseum-oran-coloran-dataset
```

Then `--data_root` defaults to `./Dataset/colosseum-oran-coloran-dataset` (one level up from `train.py`). You can always override:

```bash
python train.py --data_root /any/other/path/colosseum-oran-coloran-dataset ...
```

> Note: `mmtc` slice corresponds to the `mtc/` folder name in the dataset.

### 3. Prepare the dataset directory layout

The raw dataset is **not** in the structure `DataProcessor.py` expects
(`tr{N}/exp*/bs*/{embb|mtc|urllc}/*_metrics.csv`). Two helper scripts in
`scripts/` reshape it. **Always run with no flag first to dry-run, then add
`--execute` once the summary looks right** (the operations are destructive).

```bash
# Step 1 — keep only *_metrics.csv, move them up one level, drop empty dirs
python scripts/clean_dataset.py --dir Dataset/colosseum-oran-coloran-dataset                    # dry run
python scripts/clean_dataset.py --dir Dataset/colosseum-oran-coloran-dataset --execute          # actually run

# Step 2 — split CSVs into embb/ mtc/ urllc/ sub-folders by IMSI UE ID
python scripts/group_by_slice.py --dir Dataset/colosseum-oran-coloran-dataset                   # dry run
python scripts/group_by_slice.py --dir Dataset/colosseum-oran-coloran-dataset --execute         # actually run
```

After the two steps you should see:

```
Dataset/colosseum-oran-coloran-dataset/
└── tr0/exp1/bs1/
    ├── embb/1010123456003_metrics.csv
    ├── mtc/1010123456004_metrics.csv
    └── urllc/1010123456002_metrics.csv
```

`group_by_slice.py` reads the `slice_id` column from each CSV directly
(0 → embb, 1 → mtc, 2 → urllc) — no IMSI parsing needed.

## Quick start

Smoke test (5 epochs, small batch, CPU OK):

```bash
python train.py \
  --train_dirs tr0-9 --val_dirs tr10 --test_dirs tr15 \
  --slice_type embb \
  --epochs 5 --patience 0 --batch_size 256 \
  --learning_rate 1e-4 --device auto
```

Outputs are written to `results/<slice>_<timestamp>/` containing:

- `best_model.pt`, `last_model.pt`
- `train_log.csv`, `metrics.json`, `args.json`
- prediction plots, residual plots, anomaly overlays

## Reproducing paper results

The best per-slice configurations are listed below. Run each command individually,
or copy them into a shell script of your own.

### eMBB (R² ≈ 0.90, MSE)

```bash
python train.py \
  --train_dirs tr0-9 tr11-14 tr16-27 --val_dirs tr10 --test_dirs tr15 \
  --slice_type embb \
  --epochs 100 --patience 30 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --sequence_length 30 \
  --include_target_history true --readout mean \
  --adj_type binary_selfloop \
  --gat_head_merge mean --gat_final_head_merge mean \
  --loss_type mse --lambda_smooth 0.0 \
  --chebyshev_k 3.0 --anomaly_error_mode both
```

### mMTC (R² ≈ 0.30, Huber δ=1.0)

```bash
python train.py \
  --train_dirs tr0-9 tr11-14 tr16-27 --val_dirs tr10 --test_dirs tr15 \
  --slice_type mmtc \
  --epochs 200 --patience 30 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --sequence_length 30 \
  --include_target_history true --readout mean \
  --adj_type binary_selfloop \
  --gat_head_merge concat --gat_final_head_merge mean \
  --loss_type huber --huber_delta 1.0 --lambda_smooth 0.0 \
  --chebyshev_k 3.0 --anomaly_error_mode both
```

### uRLLC (R² ≈ 0.66, Huber δ=1.0)

```bash
python train.py \
  --train_dirs tr0-9 tr11-14 tr16-27 --val_dirs tr10 --test_dirs tr15 \
  --slice_type urllc \
  --epochs 200 --patience 30 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --sequence_length 30 \
  --include_target_history true --readout mean \
  --adj_type binary_selfloop \
  --gat_head_merge concat --gat_final_head_merge mean \
  --loss_type huber --huber_delta 1.0 --lambda_smooth 0.0 \
  --chebyshev_k 3.0 --anomaly_error_mode both
```

## Manual training commands

### Primary (paper) configuration

```bash
python train.py \
  --train_dirs tr0-24 --val_dirs tr25 tr26 --test_dirs tr27 \
  --slice_type mmtc \
  --epochs 200 --patience 30 --batch_size 1024 \
  --learning_rate 1e-4 --scheduler cosine \
  --rnn_type bilstm --sequence_length 30 \
  --include_target_history true \
  --readout mean \
  --adj_type binary_selfloop \
  --gat_head_merge concat --gat_final_head_merge mean \
  --loss_type huber --huber_delta 1.0 \
  --lambda_smooth 0.0 \
  --chebyshev_k 3.0 --anomaly_error_mode both
```

### Per-slice recommended losses

| Slice   | `--loss_type` | `--huber_delta` | Notes |
|---------|---------------|-----------------|-------|
| eMBB    | `mse`         | n/a             | already near noise floor (R² ≈ 0.90) |
| mMTC    | `huber`       | `1.0`           | breaks the MSE plateau (R² 0.08 → 0.30) |
| uRLLC   | `huber`       | `1.0`           | best for spike-periodicity (R² 0.65 → 0.66) |

`weighted_mse` is **not recommended** on this dataset (residuals blow up on outliers).

### Fairness / ablation switches

| Knob                                                          | Effect |
|----------------------------------------------------------------|--------|
| `--include_target_history false`                               | drop `sum_granted_prbs` history → N=15 (matches the model8 input width) |
| `--gat_head_merge {mean,concat}` / `--gat_final_head_merge`    | head-merge ablation (concat/mean is the paper default) |
| `--readout {mean,attention,gated}`                             | how to pool the N feature nodes into the scalar output |
| `--adj_type {binary_selfloop,binary_noselfloop,correlation}`   | adjacency variants (correlation needs `--adj_corr_threshold`) |
| `--fft_n_layers / --fft_n_heads / --fft_dim_feedforward / ...` | FFT-Transformer capacity |
| `--lambda_smooth 0.1`                                          | temporal-difference consistency regulariser (chunk-aware) |

Run `python train.py --help` for the full list.

### Trial-range syntax

`--train_dirs tr0-24 tr27` expands to `tr0, tr1, ..., tr24, tr27`. The three sets (`train_dirs`, `val_dirs`, `test_dirs`) must be disjoint or training will raise `ValueError`. When `--val_dirs` is given, `--val_split` is ignored.

## Anomaly detection

Anomaly artifacts are emitted automatically at the end of training under
`results/<run>/anomaly/`:

- `chebyshev_thresholds.json` — calibrated `μ ± kσ` from the **validation** residuals
- `flags_test.csv`           — per-timestep anomaly flags on the test set
- `overlay_test.png`         — predicted vs. ground-truth with anomaly markers

Override the threshold with:

```bash
python train.py ... --chebyshev_k 2.5 \
  --anomaly_error_mode both \
  --anomaly_extra_k 2.0 3.0 4.0
```

## Project status snapshot

| Slice  | Best loss      | Best R²  | vs. LSTM-only baseline |
|--------|----------------|---------:|------------------------:|
| eMBB   | `mse`          | 0.901    | −0.07 |
| mMTC   | `huber δ=1.0`  | **0.300** | **+0.17** |
| uRLLC  | `huber δ=1.0`  | **0.662** | **+0.57** |

## Citation

If you use this code, please cite the accompanying paper:

> *Efficient Graph Attention-based Learning for Traffic Prediction and Uncertainty-Aware Anomaly Detection in AI-driven O-RAN.*

## License

Source code is released under the MIT License (see [`LICENSE`](LICENSE)). The Colosseum O-RAN ColORAN dataset is governed by its own license (see the [dataset repository](https://github.com/wineslab/colosseum-oran-coloran-dataset)).
