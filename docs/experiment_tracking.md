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

Example: `20260616_143022_Transformer_ecg_brenton`

The experiment ID becomes the subdirectory name under `results/` and is
embedded in both JSON files. It is generated once at run start and propagated
everywhere.

---

## metadata.json

Written at **run start** (before training begins). Contains everything needed
to reproduce the experiment from scratch.

```json
{
  "experiment_id": "20260616_143022_Transformer_ecg_brenton",
  "timestamp": "2026-06-16T14:30:22Z",

  "git_commit": "abc123def456abc123def456abc123def456abc1",
  "git_branch": "main",
  "git_dirty": false,

  "config": {
    "experiment": {
      "name": "poc_ecg_transformer",
      "mode": "centralized"
    },
    "data": {
      "patient": "brenton",
      "sensor": "ecg",
      "train_condition": "resting",
      "test_condition": "post_exercise",
      "seq_len": 100,
      "step": 1,
      "train_ratio": 0.7,
      "val_ratio": 0.1
    },
    "preprocessing": {
      "target_fs": 100,
      "bandpass_low": 0.5,
      "bandpass_high": 40.0,
      "bandpass_order": 4,
      "scaler_fit_condition": "resting"
    },
    "model": {
      "name": "Transformer",
      "d_model": 64,
      "n_heads": 8,
      "e_layers": 1,
      "d_ff": 128,
      "dropout": 0.1
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
  "experiment_id": "20260616_143022_Transformer_ecg_brenton",

  "threshold": 0.0423,
  "anomaly_ratio": 1.0,

  "accuracy": 0.8712,
  "precision": 0.8234,
  "recall": 0.9102,
  "f1": 0.8648,

  "mean_train_score": 0.0181,
  "mean_test_score": 0.0534,
  "score_delta": 0.0353,
  "alert_rate": 0.3021,

  "evaluation_mode": "labeled",

  "n_train_windows": 42103,
  "n_test_windows": 18041,

  "training_time_seconds": 187.4,
  "best_epoch": 7,
  "best_val_loss": 0.00831
}
```

### Evaluation modes

**labeled**: Uses exercise condition as proxy anomaly ground truth. Computes
Accuracy, Precision, Recall, F1 via Point-Adjust protocol. POC only.

**unlabeled**: No ground truth. Reports score distribution shift and alert rate.
Intended for real deployment where no condition labels are available.

---

## FL Runs: fl_summary.json

Federated runs do not use `metadata.json` / `metrics.json` — the server has
no local data and cannot evaluate the model. Instead, `fl/server.py` writes
`fl_summary.json` to a timestamped results directory after all rounds complete.

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

  "round_history": [
    {"round": 1, "val_loss": 0.02341},
    {"round": 2, "val_loss": 0.01987},
    ...
    {"round": 10, "val_loss": 0.00891}
  ],

  "final_val_loss": 0.00891,
  "best_round": 8,
  "best_val_loss": 0.00863,
  "total_time_seconds": 312.4
}
```

The server also prints a formatted terminal summary with a bar chart of
per-round val_loss as soon as all rounds finish.

`val_loss` is a weighted average of each client's local validation reconstruction
error (MSE), weighted by number of validation samples. It measures how well the
global model reconstructs the normal baseline on held-out data — lower is better.
It does not directly give F1 or precision/recall; those require labeled evaluation
on the test set, which is a separate step (see `training/evaluator.py`).

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
# F1 scores for all Transformer experiments
jq -r '[.experiment_id, .f1] | @tsv' results/*/metrics.json | grep Transformer

# All experiments with recall > 0.9
jq -r 'select(.recall > 0.9) | .experiment_id' results/*/metrics.json

# Experiments run from a specific commit
jq -r 'select(.git_commit == "abc123") | .experiment_id' results/*/metadata.json
```

---

## Model Sweeps

When running a model sweep (from `poc_ecg_model_sweep.yaml`), each model
produces its own results directory:

```
results/
├── 20260616_143022_Transformer_ecg_brenton/
├── 20260616_143501_iTransformer_ecg_brenton/
├── 20260616_143820_PatchTST_ecg_brenton/
...
```

The sweep script writes an additional `sweep_summary.json` at the sweep level:

```json
{
  "sweep_id": "20260616_143022_sweep_ecg_brenton",
  "models": ["Transformer", "iTransformer", "PatchTST", ...],
  "best_model": "iTransformer",
  "best_f1": 0.912,
  "results": [
    {"model": "Transformer", "f1": 0.865, "experiment_id": "..."},
    ...
  ]
}
```
