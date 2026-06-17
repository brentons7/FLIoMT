"""
Dataset registry and DataLoader factory.

Maps dataset keys to Dataset classes, enabling config-driven dataset selection.

Source reference: tslib/data_provider/data_factory.py (data_dict pattern)

Usage:
    from datasets.registry import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(config)
"""

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from datasets.physio_dataset import PhysioDataset

DATASET_REGISTRY: dict[str, type] = {
    "physio": PhysioDataset,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def build_dataloaders(
    config: dict,
    partition_id: int | None = None,
    num_partitions: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test DataLoaders from an experiment config.

    Reads config sections: data, training.

    Expected config shape:
        data:
          processed_dir: data/processed      # relative to repo root
          patient: brenton
          sensor: ecg
          train_conditions: [resting]
          test_conditions: [post_exercise]
          seq_len: 100
          step: 1
          train_ratio: 0.7
          val_ratio: 0.1
        training:
          batch_size: 32
          num_workers: 0

    FL partitioning: if partition_id and num_partitions are given, the
    training DataLoader is restricted to a temporal slice of the training
    dataset. This is the "temporal" partition strategy; for other strategies,
    use fl/partition.py directly.

    Args:
        config:          Experiment config dict (from YAML)
        partition_id:    If given, apply temporal FL partitioning to train set
        num_partitions:  Total number of FL partitions

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config["data"]
    train_cfg = config.get("training", {})

    processed_dir = _REPO_ROOT / data_cfg.get("processed_dir", "data/processed")
    patient = data_cfg["patient"]
    sensor = data_cfg["sensor"]
    seq_len = data_cfg.get("seq_len", 100)
    step = data_cfg.get("step", 1)
    train_ratio = data_cfg.get("train_ratio", 0.7)
    val_ratio = data_cfg.get("val_ratio", 0.1)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    train_conditions = data_cfg.get("train_conditions", ["resting"])
    test_conditions = data_cfg.get("test_conditions", ["post_exercise"])
    # Validation uses the same conditions as training
    val_conditions = data_cfg.get("val_conditions", train_conditions)

    common_kwargs = dict(
        processed_dir=processed_dir,
        patient=patient,
        sensor=sensor,
        seq_len=seq_len,
        step=step,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_dataset = PhysioDataset(conditions=train_conditions, split="train", **common_kwargs)
    val_dataset = PhysioDataset(conditions=val_conditions, split="val", **common_kwargs)
    test_dataset = PhysioDataset(conditions=test_conditions, split="test", **common_kwargs)

    if partition_id is not None and num_partitions is not None:
        n = len(train_dataset)
        part_size = n // num_partitions
        start = partition_id * part_size
        end = start + part_size if partition_id < num_partitions - 1 else n
        train_dataset = Subset(train_dataset, list(range(start, end)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
