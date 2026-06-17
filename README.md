# FLIoMT

Federated learning for physiological anomaly detection on IoMT edge devices.

## Research Goal

FLIoMT develops a federated learning system where wearable biosensors on
discharged high-risk patients continuously monitor physiological signals. Each
patient's device trains a local anomaly detection model on that patient's
normal baseline. A federated server aggregates model updates across patients
without any raw data leaving the device.

Current stage: proof-of-concept using a single-patient ECG/PPG dataset
collected under controlled exercise conditions (resting, light activity,
post-exercise). The exercise conditions serve as a proxy for physiological
deviation in the POC evaluation.

## Sensors

| Sensor | Hardware | Signal | Channels |
|--------|----------|--------|----------|
| ECG | AD8232 + ADS1115 ADC | Cardiac electrical | 1 (voltage) |
| PPG | MAX30102 | Cardiac optical | 2 (red, IR) |

Both sensors are sampled via Raspberry Pi at ~100 Hz (software-timed).

## Approach

All models use **reconstruction-based unsupervised anomaly detection**:

1. Train on the patient's resting baseline. Loss = `MSE(x, model(x))`.
2. Score = per-timestep reconstruction error at inference.
3. Alert when score exceeds `percentile(train_scores, 100 - anomaly_ratio)`.

No labels are required at training or deployment time.

## Quick Start

### Preprocessing

```bash
# Preprocess all sessions registered in data/manifests/sessions.csv
bash scripts/preprocess.sh
```

### Centralized training

```bash
python scripts/train.py --config configs/experiments/poc_ecg_transformer.yaml
```

### Model sweep

```bash
python scripts/train.py --config configs/experiments/poc_ecg_model_sweep.yaml
```

### Federated training

The server can run on any machine (Jetson Xavier, workstation, etc.). It binds
to all interfaces and prints the IP clients should use.

```bash
# 1. Start the server (on whichever machine will aggregate):
bash scripts/fl/start_server.sh
# Prints: "Clients should connect to SERVER_IP=<ip>"

# 2. Start each client — replace <server-ip> with the IP printed above.
#    Partition IDs must be unique, 0-indexed, and < NUM_PARTITIONS.

# Jetson Nano (GPU, partition 0):
SERVER_IP=<server-ip> bash scripts/fl/start_client_nano.sh

# Raspberry Pi 5 (CPU, partition 1):
SERVER_IP=<server-ip> bash scripts/fl/start_client_pi5.sh

# Jetson Xavier as client (GPU, partition 2) — when Xavier is not the server:
SERVER_IP=<server-ip> bash scripts/fl/start_client_xavier.sh
```

Key environment overrides for the server:

```bash
ROUNDS=10 MIN_CLIENTS=2 LOCAL_EPOCHS=1 LR=0.0001 PORT=8080 bash scripts/fl/start_server.sh
```

For a 2-client run (e.g., Nano + Pi 5), pass `NUM_PARTITIONS=2` to each client
via `start_client.sh` directly, or edit the device script before running.

## Repository Structure

```
FLIoMT/
├── acquisition/          # Sensor recording scripts (run on Pi)
├── preprocessing/        # Signal processing pipelines
├── datasets/             # PyTorch Dataset classes
├── models/               # Anomaly detection model implementations
│   ├── registry.py       # Lazy model loader
│   └── layers/           # Shared layer implementations
├── training/             # Trainer, Evaluator, utilities
├── fl/                   # Flower federated learning stack
│   ├── server.py         # FedAvg aggregation server
│   ├── client.py         # PhysioAnomalyClient (fit + evaluate)
│   ├── run_client.py     # Client entry point (loads config, connects)
│   └── partition.py      # Temporal / patient / condition partitioning
├── configs/
│   ├── models/           # Per-model hyperparameter defaults
│   └── experiments/      # Full experiment configurations
├── scripts/
│   ├── train.py          # Centralized training entry point
│   ├── preprocess.sh
│   └── fl/
│       ├── start_server.sh
│       ├── start_client.sh        # Generic client (all params via env)
│       ├── start_client_nano.sh   # Jetson Nano preset
│       ├── start_client_pi5.sh    # Raspberry Pi 5 preset
│       └── start_client_xavier.sh # Jetson Xavier preset (client role)
├── data/
│   ├── raw/
│   │   ├── ecg/          # Raw ECG CSVs
│   │   └── ppg/          # Raw PPG CSVs
│   ├── processed/        # Preprocessed .npy arrays (git-ignored)
│   └── manifests/        # sessions.csv registry
└── results/              # Experiment outputs
    └── {experiment_id}/
        ├── metadata.json # Git hash + full config (git-tracked)
        └── metrics.json  # Evaluation results (git-tracked)
```

## Models

23 anomaly-detection-capable models across 3 migration tiers. All implement
`model(x: Tensor[B,L,C]) → x_hat: Tensor[B,L,C]`.

| Tier | Models | Status |
|------|--------|--------|
| 1 | Transformer, iTransformer, PatchTST, KANAD, Mamba2, Autoformer, NonStationary_Transformer, DLinear | Ported |
| 2 | TimeMixer, SegRNN, TimesNet, Informer, Crossformer, FEDformer, ETSformer, Reformer, FiLM, MICN | Partial / planned |
| 3 | MambaSimple, LightTS, Crossformer, Pyraformer, TimeFilter, MSGNet | Planned |

See `docs/model_inventory.md` for full details including layer dependencies,
edge compatibility, and notes on unsupported models.

## Federated Learning

FL uses [Flower](https://flower.ai/) with FedAvg. The server aggregates only
model weights — no patient data is transmitted.

**Partition strategies** (set in experiment config):

| Strategy | Use case |
|----------|----------|
| `temporal` | Single patient, N time windows — current POC |
| `patient` | One patient per client — multi-patient deployment |
| `condition` | One activity condition per client |

## Experiment Tracking

Every run produces `metadata.json` (written at start) and `metrics.json`
(written at end). The metadata includes:

- Git commit hash and dirty state
- Complete experiment configuration
- Python, PyTorch, Flower versions
- Platform and hostname

Results are queryable with `jq`:

```bash
# Compare F1 across all experiments
jq -r '[.experiment_id, .f1] | @tsv' results/*/metrics.json | sort -k2 -rn
```

See `docs/experiment_tracking.md` for the full schema and query patterns.

## Documentation

| Document | Contents |
|----------|----------|
| `docs/architecture.md` | System diagram, module map, anomaly detection approach |
| `docs/data_flow.md` | End-to-end data flow from sensor to metrics |
| `docs/experiment_tracking.md` | metadata.json and metrics.json schemas |
| `docs/model_inventory.md` | All models, tiers, layer deps, unsupported models |
| `results/README.md` | Results directory structure and git tracking policy |

## Source Repository

All model implementations, FL code, and training infrastructure are ported
from `../tslib/`. The tslib repository is read-only; FLIoMT is the migration
target. Do not modify tslib.

## Requirements

```bash
pip install -r requirements.txt
```

Works on any device (Xavier, Raspberry Pi, workstation). No version pins so pip
selects wheels compatible with the Python version on the target device (3.8–3.13).

Raspberry Pi sensor acquisition packages are not in requirements.txt — they only
install on Pi and are only needed for data collection:

```bash
pip install adafruit-circuitpython-ads1x15 RPi.GPIO smbus2
```

Mamba2 (Tier 1) is pure PyTorch — no additional packages, runs on all devices.
MambaSimple (Tier 3) requires the `mamba_ssm` package with CUDA.
