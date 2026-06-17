# Results Directory

Each experiment run creates a subdirectory here named by experiment ID.

## Directory Layout

```
results/
└── {experiment_id}/
    ├── metadata.json      # Git hash, timestamp, full config (git-tracked)
    ├── metrics.json       # Final evaluation metrics (git-tracked)
    ├── checkpoint.pth     # Best model weights (NOT git-tracked)
    └── scores.npy         # Per-timestep anomaly scores (NOT git-tracked)
```

## Experiment ID Format

```
{YYYYMMDD}_{HHMMSS}_{model_name}_{sensor}_{patient}
```

Example: `20260616_143022_Transformer_ecg_brenton`

## metadata.json Schema

```json
{
  "experiment_id": "20260616_143022_Transformer_ecg_brenton",
  "timestamp": "2026-06-16T14:30:22Z",
  "git_commit": "abc123def456...",
  "git_branch": "main",
  "git_dirty": false,
  "config": {
    "experiment": { ... },
    "data": { ... },
    "preprocessing": { ... },
    "model": { ... },
    "training": { ... },
    "fl": { ... }
  },
  "environment": {
    "python_version": "3.10.4",
    "torch_version": "2.0.1",
    "cuda_available": false,
    "platform": "linux",
    "hostname": "raspberrypi"
  }
}
```

## metrics.json Schema

```json
{
  "experiment_id": "20260616_143022_Transformer_ecg_brenton",
  "threshold": 0.0423,
  "accuracy": 0.8712,
  "precision": 0.8234,
  "recall": 0.9102,
  "f1": 0.8648,
  "mean_train_score": 0.0181,
  "mean_test_score": 0.0534,
  "score_delta": 0.0353,
  "alert_rate": 0.3021,
  "evaluation_mode": "labeled"
}
```

## Git Tracking Policy

Only `metadata.json` and `metrics.json` are tracked. This allows experiment
results to be compared across commits without storing large binary files.

Model weights and score arrays can be reproduced by re-running the experiment
with the config stored in `metadata.json` and the git commit recorded there.
