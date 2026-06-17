"""
Main training entry point.

Reads an experiment YAML config, builds the dataset, instantiates the model,
runs centralized training, and evaluates the result. Writes a complete
metadata.json and metrics.json to the result directory.

Usage:
    python scripts/train.py --config configs/experiments/poc_ecg_transformer.yaml

    # Run a sweep of all Tier 1 models:
    python scripts/train.py --config configs/experiments/poc_ecg_model_sweep.yaml

    # Override specific config values:
    python scripts/train.py \\
        --config configs/experiments/poc_ecg_transformer.yaml \\
        --set model.d_model=128 training.epochs=20
"""

from __future__ import annotations
import argparse
import copy
import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure repo root is on the path when running as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import build_dataloaders
from models.registry import ModelRegistry
from training.trainer import Trainer
from training.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    The config YAML is the primary source of truth. --set flags override
    individual values using dot-notation keys (e.g., model.d_model=128).
    """
    p = argparse.ArgumentParser(description="FLIoMT centralized training.")
    p.add_argument(
        "--config", required=True,
        help="Path to experiment YAML (e.g., configs/experiments/poc_ecg_transformer.yaml)"
    )
    p.add_argument(
        "--set", nargs="*", default=[],
        metavar="key=value",
        help="Override config values: --set model.d_model=128 training.epochs=20"
    )
    p.add_argument(
        "--results-dir", default=str(REPO_ROOT / "results"),
        help="Root directory for experiment results"
    )
    return p.parse_args()


def _set_nested(d: dict, key: str, value: str) -> None:
    """Set a dot-notation key in a nested dict, coercing value type."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    leaf = parts[-1]
    # Try to coerce to int, float, bool, then leave as string
    for cast in (int, float):
        try:
            d[leaf] = cast(value)
            return
        except ValueError:
            pass
    if value.lower() in ("true", "false"):
        d[leaf] = value.lower() == "true"
    else:
        d[leaf] = value


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """
    Load a YAML experiment config and apply any CLI overrides.

    Args:
        config_path: Path to experiment YAML file
        overrides:   List of "key=value" strings (dot-notation)

    Returns:
        Merged config dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"--set value must be key=value, got: {override!r}")
        key, value = override.split("=", 1)
        _set_nested(config, key.strip(), value.strip())

    return config


def _resolve_device(config: dict) -> torch.device:
    device_str = config.get("training", {}).get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _make_experiment_id(config: dict) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = config.get("model", {}).get("name", "unknown")
    sensor = config.get("data", {}).get("sensor", "unknown")
    patient = config.get("data", {}).get("patient",
              config.get("data", {}).get("patients", ["unknown"])[0])
    return f"{ts}_{model_name}_{sensor}_{patient}"


def run_experiment(config: dict, results_root: str | Path = REPO_ROOT / "results") -> dict:
    """
    Execute one training + evaluation run.

    Steps:
        1. Seed RNG
        2. Resolve device
        3. Build DataLoaders
        4. Instantiate model
        5. Train (writes metadata.json + checkpoint.pth)
        6. Evaluate (writes metrics.json)
        7. Return metrics dict

    Args:
        config:       Full resolved experiment config
        results_root: Root directory for results

    Returns:
        Metrics dict from the evaluator
    """
    # Normalize patients → patient (single patient for now)
    if "patient" not in config.get("data", {}):
        patients = config["data"].get("patients", ["brenton"])
        config["data"]["patient"] = patients[0]

    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(config)
    print(f"Device: {device}")

    experiment_id = _make_experiment_id(config)
    result_dir = Path(results_root) / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {experiment_id}")

    train_loader, val_loader, test_loader = build_dataloaders(config)

    model_name = config["model"]["name"]
    model_cls = ModelRegistry.get(model_name)
    from types import SimpleNamespace
    model_args = SimpleNamespace(**config["model"])
    model_args.seq_len = config["data"]["seq_len"]
    model_args.task_name = "anomaly_detection"
    model_args.pred_len = 0
    model_args.label_len = 0
    model = model_cls(model_args).float()
    print(f"Model: {model_name}  params={sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        result_dir=result_dir,
        device=device,
    )
    model = trainer.train()

    anomaly_ratio = config.get("training", {}).get("anomaly_ratio", 1.0)
    train_label = config["data"].get("train_conditions", ["resting"])[0]

    evaluator = Evaluator(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        anomaly_ratio=anomaly_ratio,
        device=device,
        result_dir=result_dir,
        train_label=train_label,
    )
    metrics = evaluator.run(labeled=True)
    metrics["experiment_id"] = experiment_id

    print(f"\nDone. Results in: {result_dir}")
    return metrics


def run_sweep(config: dict, results_root: str | Path = REPO_ROOT / "results") -> list[dict]:
    """
    Run a model sweep: iterate over config['sweep'] entries and call
    run_experiment for each one.

    Args:
        config:       Sweep experiment config (contains 'sweep' list)
        results_root: Root results directory

    Returns:
        List of metrics dicts, one per model
    """
    sweep_entries = config.get("sweep", [])
    if not sweep_entries:
        raise ValueError("Sweep config must contain a 'sweep' list.")

    base = {k: v for k, v in config.items() if k != "sweep"}
    all_metrics: list[dict] = []

    for entry in sweep_entries:
        run_config = copy.deepcopy(base)
        for section, values in entry.items():
            if isinstance(values, dict):
                run_config.setdefault(section, {}).update(values)
            else:
                run_config[section] = values

        model_name = run_config.get("model", {}).get("name", "unknown")
        print(f"\n{'='*60}")
        print(f"Sweep: {model_name}")
        print("=" * 60)

        try:
            metrics = run_experiment(run_config, results_root=results_root)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            all_metrics.append({"model": model_name, "error": str(e)})

    # Write sweep summary
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    sensor = base.get("data", {}).get("sensor", "unknown")
    patient = base.get("data", {}).get("patients", ["unknown"])[0]
    sweep_id = f"{ts}_sweep_{sensor}_{patient}"

    successful = [m for m in all_metrics if "f1" in m]
    best = max(successful, key=lambda m: m["f1"]) if successful else {}

    summary = {
        "sweep_id": sweep_id,
        "n_models": len(all_metrics),
        "best_model": best.get("experiment_id", ""),
        "best_f1": best.get("f1"),
        "results": all_metrics,
    }

    summary_path = Path(results_root) / f"{sweep_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSweep summary → {summary_path}")

    return all_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.set)
    results_root = args.results_dir

    if "sweep" in config:
        run_sweep(config, results_root=results_root)
    else:
        run_experiment(config, results_root=results_root)


if __name__ == "__main__":
    main()
