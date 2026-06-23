# Data Flow

## Overview

```
Raspberry Pi 5               Desktop / Edge Devices          Results
──────────────               ──────────────────────          ───────
record_ecg.py  ──CSV──►  ecg_pipeline.py  ──.npy──►  benchmark.py  ──►  benchmarks/
record_ppg.py  ──CSV──►  ppg_pipeline.py  ──.npy──►  FL client     ──►  fl_summary.json
                                                          │
                                               PhysioAnomalyClient
                                               (FL training path)
```

The primary dataset is MIT-BIH Arrhythmia Database. The acquisition pipeline
(AD8232 + MAX30102) is wired up on Pi 5 for future real-sensor data.

---

## Stage 1: Sensor Data Acquisition

**Hardware**: Raspberry Pi 5 + AD8232 (ECG via ADS1115 ADC) + MAX30102 (PPG)

**Scripts**: `acquisition/record_ecg.py`, `acquisition/record_ppg.py`

**Output format**:

ECG CSV — columns: `timestamp, raw, voltage, patient, condition`
```
2026-06-10T18:03:52.118897,12220,1.527625,brenton,resting
```

PPG CSV — columns: `timestamp, red, ir, patient, condition`
```
2026-06-10T20:24:12.555834,57900,57762,brenton,resting
```

**Sampling**: Software-timed at ~100 Hz. Realized rate is computed from
`median(diff(timestamps))` and corrected during preprocessing.

**File naming**: `{patient}_{sensor}_{condition}_{YYYYMMDD_HHMMSS}.csv`

---

## Stage 2: Preprocessing

### MIT-BIH Pipeline (`preprocessing/mitbih_pipeline.py`)

```
PhysioNet waveform record (e.g. mitbih_213)
  │
  ▼ 1. Load MLII lead via wfdb; resample to 100 Hz
  │
  ▼ 2. Separate by annotation type
  │    Normal beats → ecg_normal
  │    Arrhythmia beats (PVCs, AFib, VT, etc.) → ecg_arrhythmia
  │
  ▼ 3. Bandpass filter (Butterworth, zero-phase)
  │    low=0.5 Hz, high=40.0 Hz, order=4
  │
  ▼ 4. StandardScaler — fit on normal data only
  │    Save to data/processed/{record}/ecg_scaler.pkl
  │
  ▼ 5. Save
       data/processed/{record}/ecg_normal.npy      shape: [T, 1], float32
       data/processed/{record}/ecg_arrhythmia.npy  shape: [T, 1], float32
```

**Critical constraint**: The scaler is fit on normal data only. Fitting on
arrhythmia data would normalize away amplitude differences that are meaningful
for anomaly detection.

**Git-tracked records**: `mitbih_213` (Orin Nano #2) and `mitbih_106` (Pi 5)
are committed to the repository so edge devices can `git pull` without running
preprocessing locally.

### Personal ECG/PPG Pipeline (`preprocessing/ecg_pipeline.py`, `ppg_pipeline.py`)

Used for data collected from the AD8232 + MAX30102 on Pi 5.

```
Raw CSV [T_raw samples, irregular timestamps]
  │
  ▼ 1. Parse timestamps → compute realized_fs
  ▼ 2. Resample to regular 100 Hz grid
  ▼ 3. Bandpass filter (ECG: 0.5–40 Hz; PPG: 0.5–5 Hz)
  ▼ 4. StandardScaler — fit on normal/resting condition only
  ▼ 5. Save .npy arrays per condition
```

---

## Stage 3: Dataset Loading

**Class**: `datasets.physio_dataset.PhysioDataset`

```
data/processed/{patient}/{sensor}_{condition}.npy  [T, C]
  │
  ▼ Load .npy for each requested condition; concatenate along time axis
  ▼ Split into train / val / test by ratio
  ▼ Return (window, condition_label) via __getitem__
       window:          float32 Tensor [seq_len, C]
       condition_label: str  e.g. "normal" / "arrhythmia"
```

**Sliding window**: `window = data[i * step : i * step + seq_len]`

**Condition label use**: The label is NOT fed to the model. It is used by:
- `fl/partition.py` — assigns windows to FL client partitions
- `training/evaluator.py` — proxy anomaly label: `label = 0 if condition == train_label else 1`

---

## Stage 4A: Benchmark (No FL)

**Script**: `benchmark.py`

```
train_loader, val_loader, test_loader
  │
  ▼ For each model in ALL_MODELS:
  │   Train on train_loader (normal data)
  │   Evaluate on test_loader (arrhythmia data)
  │   Measure edge timing (CPU latency, GPU throughput)
  │
  ▼ Print ranked table (by AUROC)
  ▼ Save results/benchmarks/benchmark_{timestamp}_{patient}.json
```

Usage:
```bash
python benchmark.py --patient mitbih_213 --train-conditions normal --test-conditions arrhythmia
```

---

## Stage 4B: Federated Training

```
FL Server (Orin Nano #1 — network coordinator only)
  │
  ▼ Round 1..N:
  │   1. Broadcast global weights + fit_config to all clients
  │   2. Each client:
  │        set_parameters(model, global_weights)
  │        local SGD for local_epochs on normal (train) data
  │        return (updated_weights, n_samples, {fit_time, n_params, param_mb})
  │   3. Server: FedAvg aggregate
  │        global_weights = Σ(n_i × w_i) / Σ(n_i)
  │   4. Each client evaluates on val (normal) data:
  │        val_loss = MSE(x, model(x))
  │        return (val_loss, n_val, {val_loss, eval_time})
  │   5. Server: weighted average of val_loss (monitoring only)
  │
  ▼ Final round only — server signals is_final_round=True:
  │   Each client:
  │     Runs full Evaluator on test (arrhythmia) data
  │     Runs measure_edge() for CPU latency + GPU throughput
  │     Returns AUROC, AUPRC, F1(PA), F1(raw), score_separation, edge timing
  │
  ▼ Server saves fl_summary.json with full detection metrics
```

---

## Stage 5: Evaluation

**Class**: `training.evaluator.Evaluator`

### Labeled Mode (POC / benchmark)

```
model, train_loader (normal), test_loader (arrhythmia)
  │
  ▼ Compute train scores: E_train[i] = mean(MSE(x[i], model(x[i])))
  ▼ Compute test scores:  E_test[i]  = mean(MSE(x[i], model(x[i])))
  │   Collect binary labels: 0=normal, 1=arrhythmia
  │
  ▼ AUROC, AUPRC:
  │   Concatenate [E_train (label=0), E_test (label=gt)]
  │   → threshold-independent, primary ranking metrics
  │
  ▼ Threshold: percentile(concat(E_train, E_test), 100 − anomaly_ratio)
  │
  ▼ F1 with Point-Adjust (PA): literature standard
  ▼ F1 raw: honest baseline without PA inflation
  │
  ▼ Score statistics:
     mean_normal_score, std_normal_score, mean_anomaly_score
     score_delta, score_separation (σ)
```

### Unlabeled Mode (Real Deployment)

```
model, train_loader, deployment_loader (no labels)
  │
  ▼ Compute E_train and E_deploy
  ▼ Threshold (same formula)
  ▼ Report: alert_rate, mean scores, score_delta
```

---

## Data Schema

| Asset | Location | Format | Shape | Git-tracked |
|---|---|---|---|---|
| Raw ECG CSVs | `data/raw/ecg/` | CSV | [T, 5] | Yes |
| Raw PPG CSVs | `data/raw/ppg/` | CSV | [T, 5] | Yes |
| Session manifest | `data/manifests/sessions.csv` | CSV | rows | Yes |
| MIT-BIH normal | `data/processed/mitbih_{r}/ecg_normal.npy` | npy float32 | [T, 1] | Yes (213, 106 only) |
| MIT-BIH arrhythmia | `data/processed/mitbih_{r}/ecg_arrhythmia.npy` | npy float32 | [T, 1] | Yes (213, 106 only) |
| Scaler | `data/processed/{p}/ecg_scaler.pkl` | pickle | — | Yes (213, 106 only) |
| Other processed data | `data/processed/*/` | npy / pkl | — | No |
| Benchmark results | `results/benchmarks/*.json` | JSON | — | No |
| FL summary | `results/{id}/fl_summary.json` | JSON | — | Yes |
