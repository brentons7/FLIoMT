# Results Directory

Each experiment run creates a subdirectory here named by experiment ID.

## Directory Layout

```
results/
└── {experiment_id}/
    ├── metadata.json      # Centralized runs: git hash, timestamp, full config (git-tracked)
    ├── metrics.json       # Centralized runs: evaluation metrics (git-tracked)
    ├── fl_summary.json    # FL runs: per-round history, timing, comm overhead, detection metrics (git-tracked)
    └── checkpoint.pth     # Best model weights (NOT git-tracked)
```

## Experiment ID Format

```
Centralized:  {YYYYMMDD}_{HHMMSS}_{model}_{sensor}_{patient}
FL:           {YYYYMMDD}_{HHMMSS}_fl_{model}
Benchmark:    benchmarks/benchmark_{YYYYMMDD}_{HHMMSS}_{patient}.json
```

---

## fl_summary.json Schema

Written by the server after all rounds complete.

```json
{
  "experiment_id": "20260617_143022_fl_PatchTST",
  "timestamp": "2026-06-17T14:30:22Z",
  "config": { "model": {...}, "data": {...}, "training": {...}, "fl": {...} },

  "fl_run": {
    "rounds_completed": 10,
    "rounds_requested": 10,
    "min_clients": 2,
    "local_epochs": 1,
    "learning_rate": 0.0001
  },

  "model_size": {
    "n_params": 553216,
    "param_mb": 2.109
  },

  "timing": {
    "total_seconds": 312.4,
    "avg_round_seconds": 31.2,
    "min_round_seconds": 28.1,
    "max_round_seconds": 45.3,
    "avg_client_fit_seconds": 24.7
  },

  "communication": {
    "avg_round_mb": 2.84,
    "total_mb": 28.4
  },

  "round_history": [
    {
      "round": 1,
      "val_loss": 0.023410,
      "round_wall_seconds": 45.3,
      "comm_bytes_in": 1420000,
      "comm_bytes_out": 1420000,
      "comm_total_mb": 2.84,
      "n_clients_fit": 2,
      "client_fit_times_seconds": [38.1, 42.3],
      "avg_client_fit_seconds": 40.2,
      "client_eval_times_seconds": [2.1, 2.4],
      "avg_client_eval_seconds": 2.25
    }
  ],

  "final_val_loss": 0.008910,
  "best_round": 8,
  "best_val_loss": 0.008630,

  "detection": [
    {
      "auroc": 0.9882,
      "auprc": 0.9741,
      "f1_pa": 0.9512,
      "f1_raw": 0.8934,
      "score_separation": 15.02,
      "score_delta": 0.0631,
      "mean_normal_score": 0.0181,
      "mean_anomaly_score": 0.0812,
      "cpu_latency_ms": 0.912,
      "gpu_throughput_wps": 48200
    }
  ]
}
```

The `detection` array has one entry per client — reported on the final FL round
only. `null` if the final round produced no detection metrics.

### Field definitions

| Field | Description |
|---|---|
| `val_loss` | Weighted-average MSE reconstruction error on each client's validation (normal) set |
| `round_wall_seconds` | Server-side wall time per round: client compute + network round-trip |
| `comm_bytes_in` | Total bytes received from all clients (updated weights) |
| `comm_bytes_out` | Total bytes sent to all clients (global weights broadcast) |
| `comm_total_mb` | Bidirectional communication per round in MB |
| `client_fit_times_seconds` | Pure local training time on each device (excludes network) |
| `auroc` | Area Under ROC Curve — primary success metric |
| `auprc` | Area Under Precision-Recall Curve |
| `f1_pa` | F1 with Point-Adjust protocol (matches literature) |
| `f1_raw` | F1 without Point-Adjust (honest baseline) |
| `score_separation` | σ units between normal and arrhythmia score distributions |
| `cpu_latency_ms` | Single-window inference latency on CPU (Pi 5 scenario) |
| `gpu_throughput_wps` | Batched GPU inference throughput in windows/sec (Orin Nano scenario) |

---

## metrics.json Schema (centralized runs)

```json
{
  "experiment_id": "20260617_143022_PatchTST_ecg_mitbih_213",
  "threshold": 0.0423,
  "anomaly_ratio": 1.0,
  "auroc": 0.9882,
  "auprc": 0.9741,
  "f1_pa": 0.9512,
  "f1_raw": 0.8934,
  "mean_normal_score": 0.0181,
  "std_normal_score": 0.0042,
  "mean_anomaly_score": 0.0812,
  "score_delta": 0.0631,
  "score_separation": 15.02,
  "evaluation_mode": "labeled",
  "n_train_windows": 42103,
  "n_test_windows": 18041,
  "training_time_seconds": 187.4,
  "best_epoch": 7,
  "best_val_loss": 0.00831
}
```

---

## Git Tracking Policy

`metadata.json`, `metrics.json`, and `fl_summary.json` are git-tracked so
experiment results are versioned alongside code. Binary files (weights) are
git-ignored but reproducible from the config recorded in the JSON.

Benchmark outputs (`results/benchmarks/*.json`) are git-ignored — they are
local comparison tools, not archival results.
