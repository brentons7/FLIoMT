"""
Dataset partitioning strategies for federated learning.

Three strategies are supported:

    temporal:   Split one patient's data into N contiguous time windows.
                Non-IID because different time windows capture different
                physiological events. Used in tslib FL experiments (MSL).
                Appropriate for POC with a single patient.

    patient:    Assign each patient's complete dataset to one client.
                Most realistic for IoMT deployment. Each client holds
                truly different data (different subjects, different baselines).
                Requires multiple patients — primary strategy once additional
                patients are enrolled.

    condition:  Assign data by physiological condition (resting / light_activity
                / post_exercise). Each client trains on one condition.
                Useful for studying how model behavior generalizes across
                activity states.

Source reference: tslib/fl/partition.py (temporal only) — extended here with
patient and condition strategies and a unified make_loader dispatch.
"""

from __future__ import annotations
import math

from torch.utils.data import DataLoader, Dataset, Subset


def temporal_partition(
    dataset: Dataset,
    partition_id: int,
    num_partitions: int,
) -> Subset:
    """
    Split a dataset into N contiguous temporal slices.

    Preserves temporal order: partition 0 gets the earliest data,
    partition N-1 the most recent. Slices do not overlap.

    Source: tslib/fl/partition.py (partition_dataset)
    """
    n    = len(dataset)
    size = math.ceil(n / num_partitions)
    start = partition_id * size
    end   = min(start + size, n)
    return Subset(dataset, list(range(start, end)))


def patient_partition(
    dataset: Dataset,
    patient_id: str,
) -> Subset:
    """
    Return all windows belonging to a specific patient.

    Requires a combined multi-patient dataset where each item carries a
    patient label. PhysioDataset is single-patient; this strategy becomes
    available once multiple patients are enrolled and combined into a
    shared dataset.
    """
    raise NotImplementedError(
        "patient_partition requires a multi-patient combined dataset. "
        "Enroll additional patients and build a CombinedPhysioDataset first."
    )


def condition_partition(
    dataset: Dataset,
    conditions: list[str],
) -> Subset:
    """
    Return all windows matching the given physiological conditions.

    Scans the full dataset (O(N)) to find matching condition labels.
    PhysioDataset.__getitem__ returns (tensor, condition_str), so this
    works directly against single-patient PhysioDatasets.

    Args:
        dataset:    Dataset whose __getitem__ returns (tensor, condition_str)
        conditions: Conditions to include (e.g., ["resting", "light_activity"])
    """
    target  = set(conditions)
    indices = [i for i in range(len(dataset)) if str(dataset[i][1]) in target]
    if not indices:
        raise ValueError(
            f"No windows matched conditions {conditions}. "
            "Check that these condition names appear in the dataset."
        )
    return Subset(dataset, indices)


def make_loader(
    dataset: Dataset,
    partition_strategy: str,
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    patient_id: str | None = None,
    conditions: list[str] | None = None,
) -> DataLoader:
    """
    Build a DataLoader for a single FL client using the specified strategy.

    Source: tslib/fl/partition.py (make_loader) — extended with strategy dispatch.

    Args:
        dataset:            Full dataset to partition
        partition_strategy: "temporal", "patient", or "condition"
        partition_id:       Client index (used by temporal strategy)
        num_partitions:     Total clients (used by temporal strategy)
        batch_size:         DataLoader batch size
        num_workers:        DataLoader worker threads
        shuffle:            True for train, False for val/test
        patient_id:         Patient filter (required for patient strategy)
        conditions:         Condition list (required for condition strategy)
    """
    if partition_strategy == "temporal":
        subset = temporal_partition(dataset, partition_id, num_partitions)
    elif partition_strategy == "patient":
        if patient_id is None:
            raise ValueError("patient_id is required for patient partition strategy")
        subset = patient_partition(dataset, patient_id)
    elif partition_strategy == "condition":
        if conditions is None:
            raise ValueError("conditions is required for condition partition strategy")
        subset = condition_partition(dataset, conditions)
    else:
        raise ValueError(
            f"Unknown partition strategy '{partition_strategy}'. "
            "Choose from: temporal, patient, condition"
        )

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
