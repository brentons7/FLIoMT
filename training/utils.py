"""
Training utilities: early stopping, learning rate scheduling, anomaly
adjustment protocol, and reconstruction metrics.

Source reference: tslib/utils/tools.py, tslib/utils/metrics.py
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor validation loss and stop training when it stops improving.

    Saves the best model checkpoint whenever validation loss decreases.

    Source: tslib/utils/tools.py (EarlyStopping)

    Args:
        patience: Number of epochs to wait for improvement before stopping
        verbose:  If True, print a message each time loss improves
        delta:    Minimum change to qualify as an improvement
    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss: float, model: nn.Module, checkpoint_path: str) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str) -> None:
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model."
            )
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# ---------------------------------------------------------------------------
# Learning Rate Scheduling
# ---------------------------------------------------------------------------

def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    initial_lr: float,
    schedule: str = "type1",
    total_epochs: int = 10,
) -> None:
    """
    Adjust optimizer learning rate according to the selected schedule.

    Schedules:
        type1:  Halve LR every epoch (aggressive decay from epoch 1)
        type2:  Step decay at fixed milestones (matches tslib defaults)
        type3:  Keep initial LR for first 3 epochs, then decay 10% per epoch
        cosine: Cosine annealing from initial_lr to 0 over total_epochs

    Source: tslib/utils/tools.py (adjust_learning_rate)

    Args:
        optimizer:    PyTorch optimizer
        epoch:        Current epoch (1-indexed)
        initial_lr:   Initial learning rate from config
        schedule:     Schedule name
        total_epochs: Total training epochs (used for cosine schedule only)
    """
    if schedule == "type1":
        lr = initial_lr * (0.5 ** ((epoch - 1) // 1))
    elif schedule == "type2":
        milestones = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
        if epoch not in milestones:
            return
        lr = milestones[epoch]
    elif schedule == "type3":
        if epoch < 3:
            lr = initial_lr
        else:
            lr = initial_lr * (0.9 ** (epoch - 3))
    elif schedule == "cosine":
        lr = initial_lr / 2 * (1 + math.cos(epoch / total_epochs * math.pi))
    else:
        raise ValueError(f"Unknown LR schedule: {schedule!r}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"LR → {lr:.2e} (schedule={schedule}, epoch={epoch})")


# ---------------------------------------------------------------------------
# Point-Adjust (PA) Protocol
# ---------------------------------------------------------------------------

def adjustment(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Point-Adjust anomaly detection evaluation protocol.

    If any point within a contiguous ground-truth anomaly segment is
    correctly detected (pred=1), all points in that segment are credited
    as detected.

    Source: tslib/utils/tools.py (adjustment) — ported verbatim.

    Args:
        gt:   Ground-truth binary labels [T], 1 = anomaly
        pred: Predicted binary labels [T], 1 = detected anomaly

    Returns:
        gt, pred: gt unchanged; pred adjusted in-place and returned
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


# ---------------------------------------------------------------------------
# Edge Inference Benchmarking
# ---------------------------------------------------------------------------

def measure_edge(model: nn.Module, seq_len: int, enc_in: int) -> dict:
    """
    Measure CPU single-window latency and GPU batched throughput.

    Restores the model to its original device afterwards.
    GPU throughput is 0 when CUDA is unavailable (e.g. Pi 5).
    """
    import time
    original_device = next(model.parameters()).device
    model.eval()

    single = torch.zeros(1, seq_len, enc_in)
    cpu_m  = model.to("cpu")
    with torch.no_grad():
        for _ in range(20):
            cpu_m(single)
        N  = 500
        t0 = time.time()
        for _ in range(N):
            cpu_m(single)
    cpu_ms = round((time.time() - t0) / N * 1000, 3)

    gpu_wps = 0
    if torch.cuda.is_available():
        batched = torch.zeros(32, seq_len, enc_in).to("cuda")
        gpu_m   = model.to("cuda")
        with torch.no_grad():
            for _ in range(20):
                gpu_m(batched)
            torch.cuda.synchronize()
            N  = 200
            t0 = time.time()
            for _ in range(N):
                gpu_m(batched)
            torch.cuda.synchronize()
        gpu_wps = round(N * 32 / (time.time() - t0))

    model.to(original_device)
    return {"cpu_latency_ms": cpu_ms, "gpu_throughput_wps": gpu_wps}


# ---------------------------------------------------------------------------
# Reconstruction Metrics
# ---------------------------------------------------------------------------

def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(true - pred)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((true - pred) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def compute_reconstruction_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> dict[str, float]:
    """
    Compute MAE, MSE, and RMSE between predicted and true arrays.

    Args:
        pred: Reconstructed output, shape [T, C] or [T]
        true: Ground truth input, shape [T, C] or [T]

    Returns:
        Dict with keys "mae", "mse", "rmse"
    """
    return {"mae": mae(pred, true), "mse": mse(pred, true), "rmse": rmse(pred, true)}
