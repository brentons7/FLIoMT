# FLIoMT Architecture

## Research Goal

FLIoMT investigates federated learning for physiological anomaly detection on
IoMT (Internet of Medical Things) edge devices. The long-term objective is a
system where discharged high-risk patients continue wearing biosensors, and a
federated model — trained across multiple patients' devices without centralizing
raw data — learns to detect significant deviations from each patient's normal
physiological baseline.

Current stage: proof-of-concept validation using a single-patient ECG/PPG
dataset collected under controlled exercise conditions.

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│                   EDGE DEVICES                   │
│                                                  │
│  ┌───────────────────┐   ┌──────────────────┐    │
│  │  Orin Nano #2     │   │ Raspberry Pi 5   │    │
│  │  (partition 0)    │   │  (partition 1)   │    │
│  │  Patient Data     │   │  Patient Data    │    │
│  │  Local Model      │   │  Local Model     │    │
│  │  FL Client        │   │  FL Client       │    │
│  └────────┬──────────┘   └────────┬─────────┘    │
│           │ weights only           │ weights only  │
└───────────┼────────────────────────┼──────────────┘
            │                        │
            └────────────┬───────────┘
                         │
                         ▼ FedAvg aggregation
            ┌────────────────────────┐
            │  FL SERVER             │
            │  (Orin Nano #1)        │
            │  No local data         │
            │  Saves fl_summary.json │
            └────────────────────────┘
```

Raw sensor data never leaves the edge device. Only model weight arrays
travel over the network.

---

## Module Map

### `acquisition/`
Data collection scripts that run on Raspberry Pi devices.

- `record_ecg.py` — Samples AD8232 ECG via ADS1115 ADC at 100 Hz; writes CSV
- `record_ppg.py` — Samples MAX30102 red/IR channels at 100 Hz; writes CSV
- `monitor_ecg.py` — Live terminal BPM display during recording

### `preprocessing/`
Offline signal processing pipeline. Transforms raw CSVs into model-ready arrays.

- `ecg_pipeline.py` — Resample → bandpass (0.5–40 Hz) → StandardScaler → .npy [T,1]
- `ppg_pipeline.py` — Resample → bandpass (0.5–5 Hz) → AC/DC separation → .npy [T,2]
- `run_all.py` — Batch driver; reads `data/manifests/sessions.csv`

### `datasets/`
PyTorch Dataset classes for sliding-window training.

- `physio_dataset.py` — `PhysioDataset`: returns `(window[seq_len,C], condition_str)`
- `registry.py` — `build_dataloaders(config)`: factory for train/val/test loaders

### `models/`
Reconstruction-based anomaly detection models.

All models implement: `model(x: Tensor[B,L,C]) → x_hat: Tensor[B,L,C]`

- `registry.py` — `ModelRegistry`: lazy-loading, capability declarations, tier info
- `layers/` — Shared layer implementations (ported from tslib/layers/)
- `{model_name}.py` — Individual model implementations (23 models total)

**Model tiers:**

| Tier | Count | Status | Purpose |
|------|-------|--------|---------|
| 1    | 8     | Ported | Core experimental set; proven or high physio relevance |
| 2    | 8     | Partial / planned | Extended comparison set |
| 3    | 7     | Planned | Experimental; complex deps or niche use cases |

### `training/`
Centralized training and evaluation infrastructure.

- `trainer.py` — `Trainer`: train loop, EarlyStopping, checkpoint, metadata logging
- `evaluator.py` — `Evaluator`: anomaly scores, threshold, PA protocol, metrics
- `utils.py` — `EarlyStopping`, `adjust_learning_rate`, `adjustment`, metrics

### `fl/`
Federated learning stack using Flower (flwr).

- `server.py` — FedAvg aggregation server (network-only; no local data)
- `client.py` — `PhysioAnomalyClient(model, train_loader, val_loader, device)`
- `partition.py` — Partitioning strategies: temporal, patient-based, condition-based
- `run_client.py` — Client entry point; reads YAML config, starts Flower client

### `configs/`
YAML experiment configurations.

- `models/` — Per-model hyperparameter defaults (one file per model)
- `experiments/` — Full experiment specs: data + model + training + FL

### `scripts/`
Shell and Python entry points.

- `train.py` — Centralized training: `python scripts/train.py --config ...`
- `preprocess.sh` — Batch preprocessing wrapper
- `fl/` — Server and per-device client launcher scripts (server, client, nano, pi5, xavier)

### `data/`
- `raw/ecg/` — Raw ECG CSVs
- `raw/ppg/` — Raw PPG CSVs
- `processed/` — Preprocessed .npy arrays (git-ignored; regenerable)
- `manifests/sessions.csv` — Registry of all known sessions

### `results/`
- `{experiment_id}/metadata.json` — Git hash + full config (centralized runs; git-tracked)
- `{experiment_id}/metrics.json` — Evaluation metrics (centralized runs; git-tracked)
- `{experiment_id}/fl_summary.json` — Per-round loss history + timing (FL runs; git-tracked)
- `{experiment_id}/checkpoint.pth` — Best model weights (git-ignored)

---

## Anomaly Detection Approach

All models use **reconstruction-based unsupervised anomaly detection**:

1. **Training**: Model learns to reconstruct windows from the patient's normal baseline.
   Loss = `MSE(x, model(x))`. No labels used.

2. **Scoring**: At inference, the per-timestep reconstruction error is the anomaly score.
   `score[t] = mean_over_channels(MSE(x[t], x_hat[t]))`

3. **Thresholding**: `threshold = percentile(concat(train_scores, test_scores), 100 - anomaly_ratio)`

4. **Evaluation**:
   - **Labeled** (POC): Point-Adjust protocol → Accuracy, Precision, Recall, F1
   - **Unlabeled** (deployment): Score distribution, alert rate, threshold sensitivity

---

## Federated Learning Design Principles

1. **Server independence**: The FL server requires only a network address. It
   never accesses local data or initializes models. Orin Nano #1 runs the server
   exclusively; it holds no partition of the dataset.

2. **Client decoupling**: `PhysioAnomalyClient` accepts a pre-built model and
   pre-built DataLoaders. It has no knowledge of config files, data loading,
   or model construction.

3. **Partition strategies** (selectable via config):
   - `temporal`: One patient, N time windows — used for single-patient POC
   - `patient`: One patient per client — primary strategy for multi-patient deployment
   - `condition`: One activity condition per client — for studying condition generalization

4. **Edge compatibility**: All Tier 1 models are pure-PyTorch with no CUDA
   kernel dependencies. They run on Raspberry Pi at `batch_size=16`.

---

## Source Repository

All model implementations, FL stack, and training infrastructure are ported
from `../tslib/`. The tslib repository is read-only; FLIoMT is the migration
target. See `docs/model_inventory.md` for the complete mapping.
