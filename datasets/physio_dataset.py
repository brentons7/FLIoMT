"""
PyTorch Dataset for physiological sensor data.

Loads preprocessed .npy arrays from data/processed/ and returns fixed-length
sliding windows suitable for reconstruction-based anomaly detection.

Training is fully unsupervised: labels are not fed to the model. The
condition label is carried as metadata for evaluation and partitioning only.
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class PhysioDataset(Dataset):
    """
    Sliding-window dataset over preprocessed physiological sensor data.

    Returns (window, condition_label) pairs where:
        window:           float32 Tensor of shape [seq_len, channels]
        condition_label:  str, one of {"resting", "light_activity", "post_exercise"}

    The condition_label is NOT used during training. It is provided for:
        - Evaluation: using condition as a proxy anomaly label in the POC
        - FL partitioning: assigning windows to clients by condition

    Args:
        processed_dir:  Root of processed data (e.g., "data/processed")
        patient:        Patient ID (e.g., "brenton")
        sensor:         Sensor type: "ecg" or "ppg"
        conditions:     List of conditions to include (loaded and concatenated)
        seq_len:        Window length in samples
        step:           Step size between windows (1 = fully overlapping)
        split:          "train", "val", or "test"
        train_ratio:    Fraction of total samples used for training
        val_ratio:      Fraction of total samples used for validation
    """

    def __init__(
        self,
        processed_dir: str | Path,
        patient: str,
        sensor: str,
        conditions: list[str],
        seq_len: int = 256,
        step: int = 1,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        assert split in ("train", "val", "test"), f"split must be train/val/test, got {split!r}"
        assert sensor in ("ecg", "ppg"), f"sensor must be ecg or ppg, got {sensor!r}"

        self.seq_len = seq_len
        self.step = step
        self.split = split

        processed_dir = Path(processed_dir)
        patient_dir = processed_dir / patient

        # Load and concatenate all requested conditions
        arrays: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        for cond in conditions:
            npy_path = patient_dir / f"{sensor}_{cond}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Preprocessed file not found: {npy_path}\n"
                    f"Run: python preprocessing/run_all.py --patient {patient} --sensor {sensor}"
                )
            arr = np.load(npy_path).astype(np.float32)  # [T, C]
            arrays.append(arr)
            labels.append(np.full(len(arr), cond))

        data = np.concatenate(arrays, axis=0)    # [T_total, C]
        cond_labels = np.concatenate(labels, axis=0)  # [T_total]

        T = len(data)
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))

        if split == "train":
            self._data = data[:train_end]
            self._labels = cond_labels[:train_end]
        elif split == "val":
            self._data = data[train_end:val_end]
            self._labels = cond_labels[train_end:val_end]
        else:
            self._data = data[val_end:]
            self._labels = cond_labels[val_end:]

        # Load scaler for inverse-transform (optional; for downstream use)
        scaler_path = patient_dir / f"{sensor}_scaler.pkl"
        self._scaler: StandardScaler | None = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

        self._n_channels = data.shape[1]

    def __len__(self) -> int:
        return max(0, (len(self._data) - self.seq_len) // self.step + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        """
        Returns:
            window:    float32 Tensor [seq_len, channels]
            condition: str label for this window's source condition
        """
        start = index * self.step
        window = self._data[start : start + self.seq_len]
        condition = self._labels[start]
        return torch.from_numpy(window), condition

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def get_scaler(self) -> StandardScaler | None:
        """Return the fitted StandardScaler, or None if not found."""
        return self._scaler
