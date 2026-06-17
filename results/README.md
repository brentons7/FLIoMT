# Results Directory

Each experiment run creates a subdirectory here named by experiment ID.

## Directory Layout

```
results/
└── {experiment_id}/
    ├── metadata.json      # Centralized runs: git hash, timestamp, full config (git-tracked)
    ├── metrics.json       # Centralized runs: evaluation metrics (git-tracked)
    ├── fl_summary.json    # FL runs: per-round history, timing, comm overhead (git-tracked)
    └── checkpoint.pth     # Best model weights (NOT git-tracked)
```

## Experiment ID Format

```
Centralized:  {YYYYMMDD}_{HHMMSS}_{model}_{sensor}_{patient}
FL:           {YYYYMMDD}_{HHMMSS}_fl_{model}_{sensor}_{patient}
```

---

## fl_summary.json Schema

Written by the server after all rounds complete.

```json
{
  "experiment_id": "20260617_143022_fl_Transformer_ecg_brenton",
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
    "n_params": 123456,
    "param_mb": 0.471
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
      "n_clients_fit": 3,
      "client_fit_times_seconds": [38.1, 42.3, 35.7],
      "avg_client_fit_seconds": 38.7,
      "client_eval_times_seconds": [2.1, 2.4, 1.9],
      "avg_client_eval_seconds": 2.1
    }
  ],

  "final_val_loss": 0.008910,
  "best_round": 8,
  "best_val_loss": 0.008630
}
```

### Field definitions

| Field | Description |
|-------|-------------|
| `val_loss` | Weighted-average MSE reconstruction error on each client's validation set. Lower = model reconstructs normal signal better. |
| `round_wall_seconds` | Server-side wall time per round: client compute + network round-trip. |
| `comm_bytes_in` | Total bytes received from all clients (updated weights). |
| `comm_bytes_out` | Total bytes sent to all clients (global weights broadcast). |
| `comm_total_mb` | Bidirectional communication per round in MB. |
| `client_fit_times_seconds` | Pure local training time on each device (excludes network). |
| `client_eval_times_seconds` | Time each client spent running evaluation on its val set. |
| `n_params` | Trainable parameter count in the model. |
| `param_mb` | Model weight size in MB (float32 precision). |

---

## metrics.json Schema (centralized runs)

```json
{
  "experiment_id": "20260617_143022_Transformer_ecg_brenton",
  "threshold": 0.0423,
  "accuracy": 0.8712,
  "precision": 0.8234,
  "recall": 0.9102,
  "f1": 0.8648,
  "evaluation_mode": "labeled",
  "n_train_windows": 41696,
  "n_test_windows": 11858,
  "training_time_seconds": 101.5,
  "best_epoch": 3,
  "best_val_loss": 0.00188
}
```

---

## Git Tracking Policy

`metadata.json`, `metrics.json`, and `fl_summary.json` are git-tracked so
experiment results are versioned alongside code. Binary files (weights) are
git-ignored but reproducible from the config recorded in the JSON.
