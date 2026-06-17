# Data Flow Documentation

## Overview

```
Raspberry Pi                Local Machine / Server           Results
─────────────                ─────────────────────           ───────
record_ecg.py  ──CSV──►  ecg_pipeline.py  ──.npy──►  trainer.py  ──►  metadata.json
record_ppg.py  ──CSV──►  ppg_pipeline.py  ──.npy──►  evaluator.py ──►  metrics.json
                                                          │
                                                  PhysioAnomalyClient
                                                  (FL training path)
```

---

## Stage 1: Sensor Data Acquisition

**Hardware**: Raspberry Pi + AD8232 (ECG via ADS1115 ADC) + MAX30102 (PPG)

**Scripts**: `acquisition/record_ecg.py`, `acquisition/record_ppg.py`

**Output format**:

ECG CSV — columns: `timestamp, raw, voltage, patient, condition`
```
2026-06-10T18:03:52.118897,12220,1.527625,brenton,resting
2026-06-10T18:03:52.128946,12483,1.5605,brenton,resting
```

PPG CSV — columns: `timestamp, red, ir, patient, condition`
```
2026-06-10T20:24:12.555834,57900,57762,brenton,resting
2026-06-10T20:24:12.566113,74110,74237,brenton,resting
```

**Sampling**: Software-timed at ~100 Hz. Actual inter-sample interval varies
(Raspberry Pi is not a real-time OS). Realized rate is computed from
`median(diff(timestamps))` and must be handled in preprocessing.

**File naming**: `{patient}_{sensor}_{condition}_{YYYYMMDD_HHMMSS}.csv`

**Registration**: Each new session should be added to `data/manifests/sessions.csv`.

---

## Stage 2: Preprocessing

**Scripts**: `preprocessing/ecg_pipeline.py`, `preprocessing/ppg_pipeline.py`

### ECG Pipeline

```
Raw CSV [T_raw samples, irregular timestamps]
  │
  ▼ 1. Parse ISO 8601 timestamps → np.datetime64
  │    Compute realized_fs = 1 / median(diff(timestamps))
  │
  ▼ 2. Resample to regular grid
  │    scipy.signal.resample(voltage, int(T_raw * target_fs / realized_fs))
  │    Output: [T, 1], regular 100 Hz grid
  │
  ▼ 3. Bandpass filter (Butterworth, zero-phase)
  │    low=0.5 Hz, high=40.0 Hz, order=4
  │    Removes: DC drift, EMG artifact (>40 Hz)
  │    Preserves: QRS complex (5–40 Hz), P/T waves (0.5–10 Hz)
  │
  ▼ 4. StandardScaler
  │    Fit on the RESTING condition for this patient only.
  │    Apply to all conditions (resting, light_activity, post_exercise).
  │    Save scaler to data/processed/{patient}/ecg_scaler.pkl
  │
  ▼ 5. Save
       data/processed/{patient}/ecg_{condition}.npy   shape: [T, 1], float32
```

**Critical constraint**: The scaler is fit on resting data only. Fitting on
all conditions would normalize away the amplitude differences across conditions
that are meaningful for anomaly detection.

### PPG Pipeline

```
Raw CSV [T_raw samples, irregular timestamps, 2 channels: red, ir]
  │
  ▼ 1. Parse timestamps, compute realized_fs
  │
  ▼ 2. Resample both channels to regular grid
  │
  ▼ 3. Bandpass filter
  │    low=0.5 Hz, high=5.0 Hz (PPG cardiac band; 30–300 BPM)
  │
  ▼ 4. AC/DC separation (per channel)
  │    AC = bandpass output (pulsatile component)
  │    DC = low-pass at 0.5 Hz (slowly-varying baseline)
  │    Optional: SpO2 = 110 - 25 * (AC_red/DC_red) / (AC_ir/DC_ir)
  │
  ▼ 5. StandardScaler on AC components
  │    Fit on resting condition only.
  │
  ▼ 6. Save
       data/processed/{patient}/ppg_{condition}.npy   shape: [T, 2], float32
                                                             (red_ac, ir_ac)
```

---

## Stage 3: Dataset Loading

**Class**: `datasets.physio_dataset.PhysioDataset`

```
data/processed/{patient}/{sensor}_{condition}.npy  [T, C]
  │
  ▼ Load .npy files for requested conditions; concatenate along time axis
  │
  ▼ Compute split boundaries (train/val/test by ratio)
  │
  ▼ Return (window, condition_label) via __getitem__
       window:          float32 Tensor [seq_len, C]
       condition_label: str ("resting" / "light_activity" / "post_exercise")
```

**Sliding window**: Each call to `__getitem__(i)` returns:
- `window = data[i * step : i * step + seq_len]`
- `len(dataset) = (T - seq_len) // step + 1`

**Condition label use**: The condition label is NOT fed to the model. It is
used by:
- `fl/partition.py` — to assign windows to FL client partitions
- `training/evaluator.py` — as proxy anomaly label in labeled evaluation mode

---

## Stage 4A: Centralized Training

**Class**: `training.trainer.Trainer`

```
train_loader, val_loader
  │
  ▼ For each epoch:
  │   For each batch (x, _):          # label ignored during training
  │     x_hat = model(x)              # reconstruction
  │     loss = MSE(x, x_hat)          # reconstruction error
  │     loss.backward()
  │     optimizer.step()
  │
  ▼ EarlyStopping(patience=3) on val_loss
  │   → save checkpoint.pth when val_loss improves
  │
  ▼ Load best checkpoint
  │
  ▼ Write metadata.json (at start), metrics.json (at end)
```

---

## Stage 4B: Federated Training

```
FL Server (network coordinator)
  │
  ▼ Round 1..N:
  │   1. Broadcast global weights + fit_config to all clients
  │   2. Each client:
  │        set_parameters(model, global_weights)
  │        local SGD for local_epochs on partition data
  │        return (updated_weights, n_samples, {})
  │   3. Server: FedAvg aggregate
  │        global_weights = sum(n_i * w_i) / sum(n_i)
  │   4. Each client evaluates:
  │        val_loss = MSE(x, model(x)) on val partition
  │        return (val_loss, n_val_samples, {"val_loss": val_loss})
  │   5. Server: weighted average of val_loss (for monitoring)
  │
  ▼ After N rounds: global weights are the federated model
```

---

## Stage 5: Evaluation

**Class**: `training.evaluator.Evaluator`

### Labeled Mode (POC)

```
model, train_loader, test_loader
  │
  ▼ Compute train reconstruction energy:
  │   E_train[i] = mean_over_channels(MSE(x[i], model(x[i])))
  │   Shape: [N_train_windows]
  │
  ▼ Compute test reconstruction energy:
  │   E_test[i] = mean_over_channels(MSE(x[i], model(x[i])))
  │   Collect condition labels as proxy ground truth:
  │     label = 0 if condition == train_condition else 1
  │
  ▼ Threshold:
  │   combined = concat(E_train, E_test)
  │   threshold = percentile(combined, 100 - anomaly_ratio)
  │
  ▼ Binary predictions:
  │   pred = (E_test > threshold).astype(int)
  │
  ▼ Point-Adjust protocol (adjustment(gt, pred)):
  │   If any window in a true anomaly run is detected → credit entire run
  │
  ▼ Metrics:
     Accuracy, Precision, Recall, F1
     → write to results/{experiment_id}/metrics.json
```

### Unlabeled Mode (Real Deployment)

```
model, train_loader, deployment_loader
  │
  ▼ Compute E_train and E_deploy (no labels)
  │
  ▼ Threshold (same formula)
  │
  ▼ Report:
     alert_rate = mean(E_deploy > threshold)
     mean_train_score, mean_deploy_score, score_delta
     threshold sensitivity curve
```

---

## Data Schema Summary

| Asset | Location | Format | Shape | Git-tracked |
|-------|----------|--------|-------|-------------|
| Raw ECG CSVs | `data/ecg_raw/` | CSV | [T, 5 cols] | Yes |
| Raw PPG CSVs | `data/ppg_raw/` | CSV | [T, 5 cols] | Yes |
| Session manifest | `data/manifests/sessions.csv` | CSV | rows × cols | Yes |
| Processed ECG | `data/processed/{p}/ecg_{c}.npy` | npy float32 | [T, 1] | No |
| Processed PPG | `data/processed/{p}/ppg_{c}.npy` | npy float32 | [T, 2] | No |
| Scaler | `data/processed/{p}/ecg_scaler.pkl` | pickle | — | No |
| Model checkpoint | `results/{id}/checkpoint.pth` | torch | — | No |
| Anomaly scores | `results/{id}/scores.npy` | npy float32 | [T] | No |
| Experiment metadata | `results/{id}/metadata.json` | JSON | — | Yes |
| Evaluation metrics | `results/{id}/metrics.json` | JSON | — | Yes |
