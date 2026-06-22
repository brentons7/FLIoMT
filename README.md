# FLIoMT

Federated learning system for real-time physiological anomaly detection on IoMT edge devices. The goal is post-discharge cardiac patient monitoring: devices train locally on each patient's normal baseline and a central server aggregates model weights without raw data leaving the device.

## Hardware

| Device | Role |
|---|---|
| Orin Nano #1 | FL server — FedAvg aggregation only |
| Orin Nano #2 | FL client — CUDA training |
| Raspberry Pi 5 | FL client — CPU-only, real-time inference |

**Sensors on Pi 5:** AD8232 (ECG, 1 channel) + MAX30102 (PPG, red + IR), sampled at ~100 Hz.

## Models

Reconstruction-based unsupervised anomaly detection: train on normal signal, anomaly score = per-window MSE(x, x̂). Threshold set at `percentile(train_scores, 100 − anomaly_ratio)`.

| Model | Params | CPU ms/win | Notes |
|---|---|---|---|
| **PatchTST** | 553K | 0.9 | Primary FL candidate. AUROC 0.988 on MIT-BIH. Patch-as-token captures QRS morphology. |
| **CNNAutoencoder** | 12.6K | 0.3 | Pi 5 candidate. Fastest and smallest (0.05 MB). Dilated Conv1d, 310ms receptive field. |
| **TimesNet** | 9.4M | 15.0 | Orin Nano candidate. AUROC 0.970. 37.5 MB — too heavy for Pi 5. |
| **iTransformer** | 80K | 0.4 | Reserved for ECG+PPG multi-channel experiments (enc_in=2). |

Benchmarked on MIT-BIH record 213 (train: normal sinus, test: annotated arrhythmia). AUROC and AUPRC are the primary metrics — threshold-independent and meaningful for imbalanced detection tasks.

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
│   ├── server.py         # FedAvg aggregation server
│   ├── client.py         # PhysioAnomalyClient (fit + evaluate)
│   ├── run_client.py     # Client entry point
│   └── partition.py      # Temporal / patient / condition partitioning
├── configs/
│   ├── models/           # Per-model hyperparameter defaults
│   └── experiments/      # FL experiment configurations
├── benchmark.py          # Centralized model benchmark (no FL required)
├── data/
│   ├── raw/              # Raw sensor CSVs
│   ├── processed/        # Preprocessed .npy arrays (git-ignored)
│   └── manifests/        # sessions.csv data registry
└── results/
    └── benchmarks/       # benchmark.py JSON outputs
```

## Quick Start

**Benchmark all models (no FL required):**
```bash
source venv/bin/activate
python benchmark.py --patient mitbih_213 --train-conditions normal --test-conditions arrhythmia
```

**Run a specific model:**
```bash
python benchmark.py --models PatchTST CNNAutoencoder --patient mitbih_213 \
  --train-conditions normal --test-conditions arrhythmia --epochs 50
```

**Federated training (3 devices):**
```bash
# 1. Orin Nano #1 — start server
bash scripts/fl/start_server.sh

# 2. Orin Nano #2 — GPU client (partition 0)
SERVER_IP=<ip> bash scripts/fl/start_client_orin.sh

# 3. Raspberry Pi 5 — CPU client (partition 1)
SERVER_IP=<ip> bash scripts/fl/start_client_pi5.sh
```

Server environment overrides:
```bash
ROUNDS=50 MIN_CLIENTS=2 LOCAL_EPOCHS=1 LR=0.0001 PORT=8080 bash scripts/fl/start_server.sh
```

## Requirements

```bash
pip install -r requirements.txt
```

Works on all three devices (Python 3.8–3.13). Pi 5 sensor packages are separate:
```bash
pip install adafruit-circuitpython-ads1x15 RPi.GPIO smbus2
```
