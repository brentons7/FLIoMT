"""
Federated Learning client for physiological anomaly detection.

PhysioAnomalyClient wraps a reconstruction model and local DataLoaders
behind the Flower NumPyClient interface. Intentionally decoupled from
data loading and model construction — both are passed as arguments by
run_client.py, keeping this class testable in isolation.

Each round, fit() reports back to the server:
    fit_time_seconds  — pure local training time (excludes communication)
    eval_time_seconds — time spent in evaluate()
    n_params          — trainable parameter count
    param_mb          — model weight size in MB (float32)
"""

from __future__ import annotations
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import flwr as fl


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load a list of numpy parameter arrays into a model in-place."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def _model_stats(model: nn.Module) -> tuple[int, float]:
    """Return (n_params, size_mb) for the model's trainable parameters."""
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_params, round(size_bytes / 1e6, 4)


class PhysioAnomalyClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for reconstruction-based physiological anomaly detection.

    Each FL round:
        fit()      — receives global weights, trains locally, returns updated
                     weights + timing/size metrics
        evaluate() — receives global weights, computes val reconstruction loss
                     + evaluation time

    Args:
        model:        Instantiated reconstruction model — forward(x) -> x_hat
        train_loader: DataLoader for this client's local training partition
        val_loader:   DataLoader for this client's local validation data
        device:       Torch device for local computation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.criterion    = nn.MSELoss()

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr           = float(config.get("learning_rate", 1e-4))

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        t_fit = time.time()
        total_samples = 0
        for _ in range(local_epochs):
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.float().to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_x)
                assoc = getattr(self.model, "assoc_loss", None)
                if assoc is not None:
                    loss = loss + assoc
                loss.backward()
                optimizer.step()
                total_samples += len(batch_x)
        fit_time = time.time() - t_fit

        n_params, param_mb = _model_stats(self.model)

        return get_parameters(self.model), total_samples, {
            "fit_time_seconds": round(fit_time, 3),
            "n_params":         n_params,
            "param_mb":         param_mb,
        }

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        set_parameters(self.model, parameters)
        self.model.eval()

        t_eval = time.time()
        total_loss  = 0.0
        num_samples = 0
        with torch.no_grad():
            for batch_x, _ in self.val_loader:
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                total_loss += self.criterion(outputs, batch_x).item() * len(batch_x)
                num_samples += len(batch_x)
        eval_time = time.time() - t_eval

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return avg_loss, num_samples, {
            "val_loss":          avg_loss,
            "eval_time_seconds": round(eval_time, 3),
        }
