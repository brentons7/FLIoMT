"""
Federated Learning client entry point.

Reads an experiment YAML config, builds the model and local DataLoaders,
instantiates PhysioAnomalyClient, and connects to the FL server.

This file is the boundary between configuration and execution. All
config parsing and object construction happens here; PhysioAnomalyClient
receives ready-to-use objects with no knowledge of config or data loading.

Source reference: tslib/fl/run_client.py — replaced 130-arg argparse with
YAML config + minimal FL-specific CLI overrides.

Usage:
    python fl/run_client.py \\
        --config configs/experiments/fl_ecg_3client.yaml \\
        --server_address 192.168.1.10:8080 \\
        --partition_id 0 \\
        --num_partitions 3
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FLIoMT FL Client")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--server_address", type=str, required=True,
                        help="FL server address, e.g. 192.168.1.10:8080")
    parser.add_argument("--partition_id", type=int, default=0,
                        help="This client's partition index (0-indexed)")
    parser.add_argument("--num_partitions", type=int, default=None,
                        help="Total partitions — overrides config fl.num_partitions")
    parser.add_argument("--use_gpu", action="store_true", default=False,
                        help="Use CUDA GPU if available (default: CPU)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker threads")
    return parser.parse_args()


def _build_model_ns(config: dict) -> SimpleNamespace:
    """Merge model and data config sections into a SimpleNamespace for model __init__."""
    model_cfg = config["model"]
    data_cfg  = config["data"]
    base = dict(
        seq_len=data_cfg.get("seq_len", 100),
        enc_in=model_cfg.get("enc_in", data_cfg.get("enc_in", 1)),
        c_out=model_cfg.get("c_out",   data_cfg.get("c_out",  1)),
        d_model=model_cfg.get("d_model",  64),
        d_ff=model_cfg.get("d_ff",        32),
        n_heads=model_cfg.get("n_heads",   8),
        e_layers=model_cfg.get("e_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )
    # Pass through any remaining model keys (patch_len, stride, d_conv, etc.)
    extras = {k: v for k, v in model_cfg.items()
              if k not in ("name", *base.keys())}
    return SimpleNamespace(**base, **extras)


def main() -> None:
    args   = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    fl_cfg            = config.setdefault("fl", {})
    num_partitions    = args.num_partitions or fl_cfg.get("num_partitions", 3)
    partition_strategy = fl_cfg.get("partition_strategy", "temporal")

    train_cfg = config.setdefault("training", {})
    if args.batch_size  is not None:
        train_cfg["batch_size"]  = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers

    seed = train_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ------------------------------------------------------------------ Device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU: cuda:0")
    else:
        device = torch.device("cpu")
        if args.use_gpu:
            print("WARNING: --use_gpu set but CUDA not available; using CPU")
        else:
            print("Using CPU")

    # ------------------------------------------------------------------ Model
    model_cfg  = config["model"]
    model_name = model_cfg["name"]
    model_ns   = _build_model_ns(config)
    model_cls  = ModelRegistry.get(model_name)
    model      = model_cls(model_ns).to(device)
    n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | params: {n_params:,}")

    # --------------------------------------------------------------- DataLoaders
    batch_size  = train_cfg.get("batch_size",  16)
    num_workers = train_cfg.get("num_workers",  0)

    if partition_strategy == "temporal":
        # build_dataloaders has built-in temporal partition support
        train_loader, val_loader, _ = build_dataloaders(
            config,
            partition_id=args.partition_id,
            num_partitions=num_partitions,
        )
    else:
        # Build full unpartitioned datasets, then apply fl/partition strategy
        full_train_loader, val_loader, _ = build_dataloaders(config)
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
    print(f"Partition {args.partition_id}/{num_partitions} | strategy={partition_strategy}")
    print(f"  train={n_train} windows | val={n_val} windows | batch={batch_size}")

    # --------------------------------------------------------------- FL client
    client = PhysioAnomalyClient(model, train_loader, val_loader, device)

    print(f"Connecting to FL server at {args.server_address} ...")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
