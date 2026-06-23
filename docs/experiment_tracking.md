# Experiment Tracking

## Overview

Every training run — centralized or federated — writes two JSON files to its
results directory before any model weights are touched:

- `metadata.json`: git state, full config, environment. Written at run start.
- `metrics.json`: evaluation results. Written at run end.

These files are git-tracked. Binary outputs (weights, score arrays) are
git-ignored but are fully reproducible from `metadata.json`.

---

## Experiment ID

Format: `{YYYYMMDD}_{HHMMSS}_{model_name}_{sensor}_{patient}`

Example: `20260616_143022_PatchTST_ecg_mitbih_213`

The experiment ID becomes the subdirectory name under `results/` and is
embedded in both JSON files. It is generated once at run start and propagated
everywhere.

---

## metadata.json

Written at **run start** (before training begins). Contains everything needed
to reproduce the experiment from scratch.

```json
{
  "experiment_id": "20260616_143022_PatchTST_ecg_mitbih_213",
  "timestamp": "2026-06-16T14:30:22Z",

  "git_commit": "abc123def456abc123def456abc123def456abc1",
  "git_branch": "main",
  "git_dirty": false,

  "config": {
    "experiment": {
      "name": "poc_ecg_patchtst",
      "mode": "centralized"
    },
    "data": {
      "patient": "mitbih_213",
      "sensor": "ecg",
      "train_conditions": ["normal"],
      "test_conditions": ["arrhythmia"],
      "seq_len": 100,
      "step": 50,
      "train_ratio": 0.7,
      "val_ratio": 0.1
    },
    "preprocessing": {
      "target_fs": 100,
      "bandpass_low": 0.5,
      "bandpass_high": 40.0,
      "bandpass_order": 4,
      "scaler_fit_condition": "normal"
    },
    "model": {
      "name": "PatchTST",
      "d_model": 128,
      "n_heads": 8,
      "e_layers": 3,
      "d_ff": 256,
      "dropout": 0.1,
      "patch_len": 16,
      "stride": 8
    },
    "training": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.0001,
      "patience": 3,
      "anomaly_ratio": 1.0
    },
    "fl": null
  },

  "environment": {
    "python_version": "3.10.4",
    "torch_version": "2.0.1",
    "flwr_version": "1.5.0",
    "cuda_available": false,
    "cuda_device": null,
    "platform": "linux",
    "hostname": "raspberrypi"
  }
}
```

### Git dirty state

If `git_dirty: true`, the recorded commit hash does not fully describe the
code that ran the experiment. Treat results from dirty runs as non-reproducible
unless the diff is small and understood. For publication-quality experiments,
commit all changes before running.

---

## metrics.json

Written at **run end** (after evaluation completes).

```json
{
  "experiment_id": "20260616_143022_PatchTST_ecg_mitbih_213",

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
  "score_separation": 15.0,

  "evaluation_mode": "labeled",

  "n_train_windows": 42103,
  "n_test_windows": 18041,

  "training_time_seconds": 187.4,
  "best_epoch": 7,
  "best_val_loss": 0.00831
}
```

### Field definitions

| Field | Description |
|---|---|
| `auroc` | Area Under ROC Curve — primary ranking metric, threshold-independent |
| `auprc` | Area Under Precision-Recall Curve — preferred when anomalies are rare |
| `f1_pa` | F1 with Point-Adjust protocol (matches published literature; inflated) |
| `f1_raw` | F1 without Point-Adjust (honest baseline) |
| `mean_normal_score` | Mean reconstruction MSE on normal (train) windows |
| `std_normal_score` | Standard deviation of normal reconstruction scores |
| `mean_anomaly_score` | Mean reconstruction MSE on arrhythmia (test) windows |
| `score_delta` | `mean_anomaly_score − mean_normal_score` |
| `score_separation` | `score_delta / std_normal_score` — σ units between distributions |

### Evaluation modes

**labeled**: Uses arrhythmia condition as anomaly ground truth (MIT-BIH
annotation-derived). Computes AUROC, AUPRC, F1(PA), F1(raw). POC and
benchmark mode.

**unlabeled**: No ground truth. Reports score distribution shift and alert rate.
Intended for real deployment where no condition labels are available.

---

## FL Runs: fl_summary.json

Federated runs do not use `metadata.json` / `metrics.json` — the server has
no local data and cannot evaluate the model. Instead, `fl/server.py` writes
`fl_summary.json` to a timestamped results directory after all rounds complete.

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
      "comm_total_mb": 2.84,
      "n_clients_fit": 2,
      "client_fit_times_seconds": [38.1, 42.3],
      "avg_client_fit_seconds": 40.2,
      "client_eval_times_seconds": [2.1, 2.4],
      "avg_client_eval_seconds": 2.25
    },
    ...
    {
      "round": 10,
      "val_loss": 0.008910
    }
  ],

  "final_val_loss": 0.008910,
  "best_round": 8,
  "best_val_loss": 0.008630,

  "detection": [
    {
      "val_loss": 0.008910,
      "eval_time_seconds": 2.1,
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
    },
    {
      "auroc": 0.9743,
      "auprc": 0.9610,
      "f1_pa": 0.9301,
      "f1_raw": 0.8712,
      "score_separation": 12.8,
      "cpu_latency_ms": 0.934,
      "gpu_throughput_wps": 0
    }
  ]
}
```

The `detection` array has one entry per client (in the order clients reported).
It is `null` if no clients returned detection metrics (final round failure).

`val_loss` throughout is a weighted average of each client's local validation
reconstruction error (MSE). It measures how well the global model reconstructs
the normal baseline on held-out data — lower is better. AUROC and AUPRC are
the primary success metrics.

---

## Implementation: metadata capture in code

`training/trainer.py` — `_write_metadata()` method:

```python
import git
import torch
import platform
import datetime

def _write_metadata(self):
    repo = git.Repo(search_parent_directories=True)
    meta = {
        "experiment_id": self.experiment_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": repo.head.commit.hexsha,
        "git_branch": repo.active_branch.name,
        "git_dirty": repo.is_dirty(),
        "config": self.config,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else None,
            "platform": platform.system().lower(),
            "hostname": platform.node(),
        }
    }
    out = self.results_dir / "metadata.json"
    out.write_text(json.dumps(meta, indent=2))
```

The `gitpython` package is in `requirements.txt`. If the script is run outside
a git repository (e.g., bare deployment), wrap the `git.Repo()` call in a
try/except and record `"git_commit": null`.

---

## Querying Results

To compare experiments across runs from the shell:

```bash
# AUROC scores for all PatchTST experiments
jq -r '[.experiment_id, .auroc] | @tsv' results/*/metrics.json | grep PatchTST

# All experiments with AUROC > 0.95
jq -r 'select(.auroc > 0.95) | .experiment_id' results/*/metrics.json

# FL detection metrics from all runs
jq -r '.detection[]? | [.auroc, .auprc, .cpu_latency_ms] | @tsv' results/*/fl_summary.json

# Experiments run from a specific commit
jq -r 'select(.git_commit == "abc123") | .experiment_id' results/*/metadata.json
```
