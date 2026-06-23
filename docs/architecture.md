# FLIoMT Architecture

## Research Goal

FLIoMT investigates federated learning for physiological anomaly detection on IoMT edge devices. The objective is post-discharge cardiac patient monitoring: devices train locally on each patient's normal ECG/PPG baseline and a federated server aggregates model weights without raw data leaving the device.

Current dataset: MIT-BIH Arrhythmia Database — train on normal sinus rhythm, test on annotated arrhythmia beats.

---

## System Architecture

```
┌──────────────────────────────────────────────────┐
│                   EDGE DEVICES                   │
│                                                  │
│  ┌───────────────────┐   ┌──────────────────┐    │
│  │  Orin Nano #2     │   │ Raspberry Pi 5   │    │
│  │  mitbih_213       │   │  mitbih_106      │    │
│  │  CUDA training    │   │  CPU-only        │    │
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

Raw sensor data never leaves the edge device. Only model weight arrays travel over the network.

---

## Module Map

### `acquisition/`
Data collection scripts that run on Raspberry Pi 5.

- `record_ecg.py` — Samples AD8232 ECG via ADS1115 ADC at ~100 Hz; writes CSV
- `record_ppg.py` — Samples MAX30102 red/IR channels at ~100 Hz; writes CSV
- `monitor_ecg.py` — Live terminal BPM display during recording

### `preprocessing/`
Offline signal processing pipeline. Transforms raw CSVs into model-ready arrays.

- `ecg_pipeline.py` — Resample → bandpass (0.5–40 Hz) → StandardScaler → .npy [T,1]
- `ppg_pipeline.py` — Resample → bandpass (0.5–5 Hz) → AC/DC separation → .npy [T,2]
- `run_all.py` — Batch driver; reads `data/manifests/sessions.csv`
- `mitbih_pipeline.py` — Preprocesses MIT-BIH PhysioNet records into ecg_normal.npy / ecg_arrhythmia.npy

### `datasets/`
PyTorch Dataset classes for sliding-window training.

- `physio_dataset.py` — `PhysioDataset`: returns `(window[seq_len,C], condition_str)`
- `registry.py` — `build_dataloaders(config)`: factory returning (train_loader, val_loader, test_loader)

### `models/`
Reconstruction-based anomaly detection models. All implement:
`model(x: Tensor[B,L,C]) → x_hat: Tensor[B,L,C]`

- `registry.py` — `ModelRegistry`: lazy-loading dict; `.get(name)` returns the class
- `layers/` — Shared layer implementations (Embed, Attention, Conv, Norm, etc.)

**Active models:**

| Model | Params | CPU ms/win | Role |
|---|---|---|---|
| `PatchTST.py` | 553K | 0.9 | Primary FL candidate — AUROC 0.988 on MIT-BIH |
| `CNNAutoencoder.py` | 12.6K | 0.3 | Pi 5 candidate — fastest, 0.05 MB |
| `TimesNet.py` | 9.4M | 15.0 | Orin Nano candidate — AUROC 0.970, too heavy for Pi 5 |
| `iTransformer.py` | 80K | 0.4 | Reserved for ECG+PPG multi-channel (enc_in=2) |

### `training/`
Training and evaluation infrastructure.

- `trainer.py` — `Trainer`: train loop, EarlyStopping, checkpoint, metadata logging
- `evaluator.py` — `Evaluator`: anomaly scores, threshold, AUROC/AUPRC, PA protocol, score statistics
- `utils.py` — `EarlyStopping`, `adjust_learning_rate`, `adjustment`, `measure_edge`

### `fl/`
Federated learning stack using Flower (flwr).

- `server.py` — `TimedFedAvg`: FedAvg + per-round timing, communication tracking, final-round detection metric collection
- `client.py` — `PhysioAnomalyClient(model, train_loader, val_loader, device, test_loader, train_label, seq_len, enc_in)`: on the final FL round, runs full Evaluator + edge timing and returns detection metrics to the server
- `partition.py` — Partitioning strategies: temporal, patient-based, condition-based
- `run_client.py` — Client entry point; builds model and dataloaders, starts Flower client

### `configs/`
- `models/` — Per-model hyperparameter defaults (YAML)
- `experiments/` — Full FL experiment configs (data + model + training + FL)

### `scripts/`
- `benchmark.py` — Centralized model benchmark — no FL required; trains and evaluates all models on the same data, prints ranked tables, saves JSON to `results/benchmarks/`
- `scripts/fl/` — Server and per-device client launchers (`start_server.sh`, `start_client_orin.sh`, `start_client_pi5.sh`)

### `data/`
- `raw/ecg/` — Raw ECG CSVs from AD8232
- `raw/ppg/` — Raw PPG CSVs from MAX30102
- `processed/mitbih_213/` — Preprocessed MIT-BIH record 213 (Orin Nano #2 patient, git-tracked)
- `processed/mitbih_106/` — Preprocessed MIT-BIH record 106 (Pi 5 patient, git-tracked)
- `manifests/sessions.csv` — Registry of all known sessions

### `results/`
- `benchmarks/` — benchmark.py JSON outputs
- `{experiment_id}/fl_summary.json` — Per-round loss history, timing, comm overhead, and final-round detection metrics

---

## Anomaly Detection Approach

All models use **reconstruction-based unsupervised anomaly detection**:

1. **Training**: Model learns to reconstruct windows from the patient's normal baseline.
   Loss = `MSE(x, model(x))`. No labels used.

2. **Scoring**: Per-window reconstruction error is the anomaly score.
   `score[i] = mean(MSE(x[i], x_hat[i]))` over (seq_len × channels)

3. **Thresholding**: `threshold = percentile(concat(train_scores, test_scores), 100 − anomaly_ratio)`

4. **Evaluation** (labeled mode):
   - AUROC, AUPRC — threshold-independent, primary ranking metrics
   - F1 with Point-Adjust (PA) — matches published literature
   - F1 raw — honest baseline without PA inflation
   - Score separation (σ) — how many normal-score standard deviations separate the two distributions

---

## Federated Learning Design

1. **Server independence**: The FL server requires only a network address. It never accesses data or initializes models. Orin Nano #1 runs the server exclusively.

2. **Client decoupling**: `PhysioAnomalyClient` accepts a pre-built model and pre-built DataLoaders. It has no knowledge of config files or model construction.

3. **Final-round evaluation**: On the last FL round, each client runs the full Evaluator on its test set (arrhythmia data) and the `measure_edge()` timing benchmark. These metrics are returned to the server and saved in `fl_summary.json` under `"detection"`.

4. **Partition strategies** (selectable via config):
   - `temporal`: One patient, N time windows — single-patient POC
   - `patient`: One patient per client — multi-patient deployment target
   - `condition`: One activity condition per client
