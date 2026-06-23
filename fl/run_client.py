"""
Federated Learning client entry point.

Builds the model and local DataLoaders, then connects to the FL server.
All parameters can be passed as CLI args — no YAML config required.
A --config YAML can still be provided and will be used as a base, with any
CLI args taking precedence over it.

Usage (no YAML):
    python fl/run_client.py \\
        --server_address 192.168.1.10:8080 \\
        --model PatchTST \\
        --patient wesad_S2 \\
        --partition_id 0 --num_partitions 1

Usage (YAML base + overrides):
    python fl/run_client.py \\
        --config configs/experiments/fl_wesad_2client.yaml \\
        --server_address 192.168.1.10:8080 \\
        --partition_id 0
"""

from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
import flwr as fl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.registry import ModelRegistry
from datasets.registry import build_dataloaders
from fl.client import PhysioAnomalyClient
from fl.partition import make_loader


# Default architecture params per model — used when no YAML is provided.
# CLI args always override these.
_MODEL_PRESETS: dict[str, dict] = {
    # iTransformer: inverted-attention treats channels as tokens.
    # With 1 ECG channel the attention is a 1×1 no-op; increased depth and
    # flat LR give the FFN layers more capacity and gradient steps to compensate.
    "iTransformer": {
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 3, "dropout": 0.1,
    },
    # PatchTST: e_layers=4 (one deeper than baseline run) at seq_len=128
    # gives 15 non-overlapping patches; d_model=128 is the proven sweet spot.
    "PatchTST": {
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 4,
        "dropout": 0.1, "patch_len": 16, "stride": 8,
    },
    "TimesNet": {
        "d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "top_k": 5, "num_kernels": 6,
    },
    # CNNAutoencoder: e_layers=5 extends dilated-conv receptive field from
    # ~310 ms (4 layers) to ~630 ms (5 layers) — one full cardiac cycle at 70 bpm.
    "CNNAutoencoder": {
        "d_model": 32, "d_ff": 64, "n_heads": 1, "e_layers": 5,
        "dropout": 0.1,
    },
}
_DEFAULT_ARCH: dict = {"d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 2, "dropout": 0.1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLIoMT FL Client")

    # ── Config (optional) ─────────────────────────────────────────────────────
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config; CLI args override any YAML values")

    # ── FL ────────────────────────────────────────────────────────────────────
    p.add_argument("--server_address", type=str, required=True,
                   help="FL server address, e.g. 192.168.1.10:8080")
    p.add_argument("--partition_id",   type=int, default=0)
    p.add_argument("--num_partitions", type=int, default=None)

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--use_gpu",    action="store_true", default=False)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--num_workers",type=int,   default=None)

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--patient", type=str, default=None,
                   help="Patient/subject ID, e.g. wesad_S2")
    p.add_argument("--sensor",  type=str, default=None,
                   help="Sensor type: ecg | ppg (default: ecg)")
    p.add_argument("--seq_len", type=int, default=None,
                   help="Sequence window length (default: 128)")

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model",   type=str, default=None,
                   help="Model name from registry: PatchTST, CNNAutoencoder, TimesNet, iTransformer")

    # ── Architecture overrides ────────────────────────────────────────────────
    p.add_argument("--d_model",      type=int,   default=None)
    p.add_argument("--d_ff",         type=int,   default=None)
    p.add_argument("--n_heads",      type=int,   default=None)
    p.add_argument("--e_layers",     type=int,   default=None)
    p.add_argument("--dropout",      type=float, default=None)
    p.add_argument("--patch_len",    type=int,   default=None, help="PatchTST")
    p.add_argument("--stride",       type=int,   default=None, help="PatchTST")
    p.add_argument("--top_k",        type=int,   default=None, help="TimesNet")
    p.add_argument("--num_kernels",  type=int,   default=None, help="TimesNet")

    return p.parse_args()


def _build_config(args: argparse.Namespace) -> dict:
    """Build a config dict from CLI args, optionally merged on top of a YAML base."""
    config: dict = {}

    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # Resolve model name (CLI > YAML > default)
    model_name = args.model or config.get("model", {}).get("name", "PatchTST")

    # Architecture: defaults → preset → YAML model section → CLI overrides
    arch = {**_DEFAULT_ARCH, **_MODEL_PRESETS.get(model_name, {})}
    arch.update({k: v for k, v in config.get("model", {}).items()
                 if k not in ("name",) and v is not None})
    _ARCH_ARGS = ("d_model", "d_ff", "n_heads", "e_layers", "dropout",
                  "patch_len", "stride", "top_k", "num_kernels", "moving_avg")
    for key in _ARCH_ARGS:
        val = getattr(args, key, None)
        if val is not None:
            arch[key] = val

    config.setdefault("model", {})
    config["model"] = {"name": model_name, "enc_in": 1, "c_out": 1, **arch}

    # Data section
    config.setdefault("data", {})
    data = config["data"]
    if args.patient  is not None: data["patient"]  = args.patient
    if args.sensor   is not None: data["sensor"]   = args.sensor
    if args.seq_len  is not None: data["seq_len"]  = args.seq_len
    data.setdefault("patient",           "mitbih_213")
    data.setdefault("sensor",            "ecg")
    data.setdefault("seq_len",           128)
    data.setdefault("step",              data["seq_len"] // 2)
    data.setdefault("train_conditions",  ["normal"])
    data.setdefault("val_conditions",    ["normal"])
    data.setdefault("test_conditions",   ["arrhythmia"])
    data.setdefault("train_ratio",       0.7)
    data.setdefault("val_ratio",         0.1)

    # Preprocessing defaults
    config.setdefault("preprocessing", {
        "target_fs":            100,
        "ecg":                  {"bandpass_low": 0.5, "bandpass_high": 40.0, "filter_order": 4},
        "scaler":               "standard",
        "scaler_fit_condition": "baseline",
    })

    # Training defaults
    config.setdefault("training", {})
    train = config["training"]
    if args.batch_size  is not None: train["batch_size"]  = args.batch_size
    if args.num_workers is not None: train["num_workers"] = args.num_workers
    train.setdefault("batch_size",  16)
    train.setdefault("num_workers", 0)
    train.setdefault("seed",        42)

    # FL defaults
    config.setdefault("fl", {"enabled": True})
    config["fl"].setdefault("partition_strategy", "temporal")
    if args.num_partitions is not None:
        config["fl"]["num_partitions"] = args.num_partitions
    config["fl"].setdefault("num_partitions", 1)

    config.setdefault("experiment", {"name": "fl_run", "mode": "federated"})

    return config


def _build_model_ns(config: dict) -> SimpleNamespace:
    model_cfg = config["model"]
    data_cfg  = config["data"]
    base = dict(
        seq_len  = data_cfg.get("seq_len", 100),
        enc_in   = model_cfg.get("enc_in", data_cfg.get("enc_in", 1)),
        c_out    = model_cfg.get("c_out",  data_cfg.get("c_out",  1)),
        d_model  = model_cfg.get("d_model",  64),
        d_ff     = model_cfg.get("d_ff",    128),
        n_heads  = model_cfg.get("n_heads",   8),
        e_layers = model_cfg.get("e_layers",  2),
        dropout  = model_cfg.get("dropout", 0.1),
    )
    extras = {k: v for k, v in model_cfg.items()
              if k not in ("name", *base.keys())}
    return SimpleNamespace(**base, **extras)


def main() -> None:
    args   = parse_args()
    config = _build_config(args)

    fl_cfg             = config["fl"]
    num_partitions     = fl_cfg.get("num_partitions", 1)
    partition_strategy = fl_cfg.get("partition_strategy", "temporal")

    train_cfg = config["training"]
    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU: cuda:0")
    else:
        device = torch.device("cpu")
        if args.use_gpu:
            print("WARNING: --use_gpu set but CUDA not available; using CPU")
        else:
            print("Using CPU")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = config["model"]["name"]
    model_ns   = _build_model_ns(config)
    model_cls  = ModelRegistry.get(model_name)
    model      = model_cls(model_ns).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | params: {n_params:,}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    batch_size  = train_cfg.get("batch_size",  16)
    num_workers = train_cfg.get("num_workers",  0)

    if partition_strategy == "temporal":
        train_loader, val_loader, test_loader = build_dataloaders(
            config,
            partition_id=args.partition_id,
            num_partitions=num_partitions,
        )
    else:
        full_train_loader, val_loader, test_loader = build_dataloaders(config)
        train_loader = make_loader(
            full_train_loader.dataset,
            partition_strategy=partition_strategy,
            partition_id=args.partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            patient_id=fl_cfg.get("patient_id"),
            conditions=fl_cfg.get("conditions"),
        )

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"Partition {args.partition_id + 1}/{num_partitions} | strategy={partition_strategy}")
    print(f"  train={n_train} windows | val={n_val} windows | batch={batch_size}")

    # ── FL client ─────────────────────────────────────────────────────────────
    data_cfg  = config["data"]
    model_cfg = config["model"]
    client = PhysioAnomalyClient(
        model, train_loader, val_loader, device,
        test_loader=test_loader,
        train_label=data_cfg["train_conditions"][0],
        seq_len=data_cfg.get("seq_len", 100),
        enc_in=model_cfg.get("enc_in", 1),
    )

    print(f"Connecting to FL server at {args.server_address} ...")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
