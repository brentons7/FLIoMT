# FLIoMT

Federated learning system for physiological anomaly detection on IoMT edge devices. The goal is post-discharge cardiac patient monitoring: each device trains locally on a patient's normal baseline and a central server aggregates model weights using FedAvg — raw data never leaves the device.

Anomaly detection is reconstruction-based and unsupervised: models are trained on normal signal only. Anomaly score = per-window MSE(x, x̂). Threshold = `percentile(train_scores, 100 − anomaly_ratio)`.

## Hardware

| Device | Role | Notes |
|--------|------|-------|
| Orin Nano #1 | FL server — FedAvg aggregation only | No local data |
| Orin Nano #2 | FL client 0 — CUDA training | MIT-BIH record 213 |
| Raspberry Pi 5 | FL client 1 — CPU inference | MIT-BIH record 106 |

**Sensors on Pi 5:** AD8232 (ECG, 1 ch) + MAX30102 (PPG, red + IR), sampled at 100 Hz.

## Models

Three architectures compared on MIT-BIH Arrhythmia Database (train: normal sinus rhythm, test: annotated arrhythmia). Client 0 = Orin Nano #2, Client 1 = Pi 5. All F1 (Point-Adjust) = 1.0 / 1.0 for all three.

| Model | Params | AUROC (c0 / c1) | AUPRC (c0 / c1) | CPU ms (c0 / c1) | Comm/run |
|-------|--------|-----------------|-----------------|------------------|----------|
| **iTransformer** | 430K | **0.9989 / 0.9946** | **0.9912 / 0.9674** | 3.7 / 4.2 | 1.38 GB |
| CNNAutoencoder | 18.9K | 0.962 / 0.990 | 0.887 / 0.945 | 7.4 / **1.1** | **33 MB** |
| PatchTST | 794K | 0.929 / 0.981 | 0.810 / 0.922 | 8.9 / 16.2 | 3.44 GB |

**TimesNet** was evaluated but excluded: 9.4M params, 350 ms CPU inference on Pi 5, 15.1 GB communication over 100 rounds. Not viable for edge IoMT deployment.

### Per-model architecture (tuned)

| Param | iTransformer | CNNAutoencoder | PatchTST |
|-------|-------------|----------------|----------|
| d_model | 128 | 32 | 128 |
| e_layers | 3 | 6 | 4 |
| d_ff | 256 | — | 256 |
| n_heads | 8 | — | 8 |
| patch_len / stride | — | — | 16 / 8 |
| RF at 100 Hz | — | ~1.27 s | — |

### Per-model training config (tuned)

| Param | iTransformer | CNNAutoencoder | PatchTST |
|-------|-------------|----------------|----------|
| rounds | 200 | 100 | 150 |
| local_epochs | 2 | 1 | 1 |
| learning_rate | 1e-4 (flat) | 1e-4 → 1e-5 cosine | 1e-4 → 1e-5 cosine |
| seq_len | 128 | 128 | 128 |
| batch_size (Orin / Pi 5) | 32 / 16 | 64 / 32 | 32 / 16 |

## Experiment History

Two rounds of experiments were run on MIT-BIH. The baseline used identical hyperparameters across all models; the tuned run gave each model per-model-optimal settings.

### Round 1 — Baseline (uniform hyperparameters)

100 rounds, local_epochs=1, cosine LR 1e-4→1e-5, seq_len=100, batch_size=16.

| Model | Params | AUROC (c0 / c1) | F1_PA (c0 / c1) | Time | Comm |
|-------|--------|-----------------|-----------------|------|------|
| PatchTST | 553K | 0.979 / 0.897 | 1.0 / 1.0 | 21 min | 1.9 GB |
| CNNAutoencoder | 12.6K | 0.863 / 0.991 | 1.0 / 1.0 | 5 min | 22 MB |
| iTransformer | 80K | 0.611 / **0.451** | 1.0 / **0.0** | 7 min | 130 MB |
| TimesNet | 9.4M | 0.779 / 0.965 | 1.0 / 1.0 | 56 min | 15.1 GB |

iTransformer failed on client 1 (AUROC below chance, detected zero anomaly windows). Root cause: cosine LR schedule decayed to near-zero before the model converged, combined with insufficient rounds and local epochs. TimesNet excluded from further runs.

### Round 2 — Tuned (per-model optimal hyperparameters)

Each model trained until convergence with architecture and schedule suited to its dynamics.

| Model | Params | AUROC (c0 / c1) | F1_PA (c0 / c1) | Time | Comm |
|-------|--------|-----------------|-----------------|------|------|
| iTransformer | 430K | **0.9989 / 0.9946** | 1.0 / 1.0 | 16 min | 1.38 GB |
| CNNAutoencoder | 15.7K | 0.959 / 0.970 | 1.0 / 1.0 | 3.5 min | 27 MB |
| PatchTST | 794K | 0.929 / 0.981 | 1.0 / 1.0 | 26 min | 3.44 GB |

iTransformer reversed completely — from worst to best. With flat LR, 200 rounds, and 2 local epochs per round, the FFN stack learns excellent reconstruction despite the inverted-attention mechanism being non-functional for single-channel ECG (1×1 attention is a no-op; capacity resides entirely in the FFN layers). CNNAutoencoder extended e_layers 4→5 closed the gap on client 0 (0.863→0.959) while remaining the most communication-efficient model by far.

## Running Experiments

### Benchmark all models (no FL, no extra devices)

```bash
source venv/bin/activate
python benchmark.py --patient mitbih_213 --train-conditions normal --test-conditions arrhythmia
```

### Federated training — tuned per-model scripts

Three per-model server scripts encode the optimal hyperparameters. Start the server first, then launch both clients.

**iTransformer** (best detection):
```bash
# Orin #1
bash scripts/fl/start_server_itransformer.sh

# Orin #2
SERVER_IP=<ip> MODEL=iTransformer BATCH_SIZE=32 bash scripts/fl/start_client_orin.sh

# Pi 5
SERVER_IP=<ip> MODEL=iTransformer BATCH_SIZE=16 bash scripts/fl/start_client_pi5.sh
```

**CNNAutoencoder** (best edge efficiency):
```bash
# Orin #1
bash scripts/fl/start_server_cnn.sh

# Orin #2
SERVER_IP=<ip> MODEL=CNNAutoencoder BATCH_SIZE=64 bash scripts/fl/start_client_orin.sh

# Pi 5
SERVER_IP=<ip> MODEL=CNNAutoencoder BATCH_SIZE=32 bash scripts/fl/start_client_pi5.sh
```

**PatchTST**:
```bash
# Orin #1
bash scripts/fl/start_server_patchtst.sh

# Orin #2
SERVER_IP=<ip> MODEL=PatchTST BATCH_SIZE=32 bash scripts/fl/start_client_orin.sh

# Pi 5
SERVER_IP=<ip> MODEL=PatchTST BATCH_SIZE=16 bash scripts/fl/start_client_pi5.sh
```

### Manual env overrides

Any variable can be overridden at the call site:
```bash
ROUNDS=50 LR=0.0005 LR_SCHEDULE=none bash scripts/fl/start_server.sh
```

## Repository Structure

```
FLIoMT/
├── acquisition/          # Sensor recording scripts (run on Pi 5)
├── preprocessing/        # ECG/PPG signal processing pipelines
├── datasets/             # PyTorch Dataset and DataLoader builders
├── models/               # Model implementations
│   ├── registry.py       # Lazy model loader
│   └── layers/           # Shared layers (Embed, Attention, Conv, etc.)
├── training/             # Trainer, Evaluator, EarlyStopping, utilities
├── fl/                   # Flower federated learning stack
│   ├── server.py         # FedAvg aggregation server with per-round comm/timing tracking
│   ├── client.py         # PhysioAnomalyClient (fit + evaluate)
│   ├── run_client.py     # Client entry point — model presets and data loading
│   └── partition.py      # Temporal / patient / condition partitioning
├── configs/
│   ├── models/           # Per-model tuned architecture defaults
│   └── experiments/      # Dataset and FL experiment reference configs
├── scripts/fl/
│   ├── start_server.sh              # Generic server (accepts env var overrides)
│   ├── start_server_patchtst.sh     # PatchTST — 150 rounds, cosine LR
│   ├── start_server_cnn.sh          # CNNAutoencoder — 100 rounds, cosine LR
│   ├── start_server_itransformer.sh # iTransformer — 200 rounds, flat LR
│   ├── start_client_orin.sh         # Orin Nano #2 client defaults
│   ├── start_client_pi5.sh          # Pi 5 client defaults
│   └── start_client.sh              # Generic client launcher
├── benchmark.py          # Centralized model benchmark (no FL required)
├── data/
│   ├── raw/              # Raw sensor CSVs
│   ├── processed/        # Preprocessed .npy arrays (git-ignored)
│   └── manifests/        # sessions.csv data registry
└── results/              # Per-run experiment outputs (fl_summary.json, git-tracked)
```

## Requirements

```bash
pip install -r requirements.txt
```

Works on Python 3.10+ across all three devices. Pi 5 sensor packages:
```bash
pip install adafruit-circuitpython-ads1x15 RPi.GPIO smbus2
```
