"""
Centralized training loop for reconstruction-based anomaly detection.

Models are trained as autoencoders: given an input window x of shape
[batch, seq_len, channels], the model produces a reconstruction x_hat
of the same shape. MSE(x, x_hat) is the training loss. No labels are used.

Source reference: tslib/exp/exp_anomaly_detection.py (train method)
"""

from __future__ import annotations
import datetime
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from training.utils import EarlyStopping, adjust_learning_rate


class Trainer:
    """
    Reconstruction-based anomaly detection trainer.

    Handles: optimizer setup, training loop, validation, early stopping,
    learning rate scheduling, checkpoint saving, and metadata logging.

    Args:
        config:       Full experiment config dict (from YAML)
        model:        Instantiated PyTorch model: model(x) → x_hat
        train_loader: DataLoader for training windows
        val_loader:   DataLoader for validation windows
        result_dir:   Directory where checkpoints and metadata are saved
        device:       torch.device to use
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        result_dir: str | Path,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.result_dir = Path(result_dir)
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 10)
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.patience = train_cfg.get("patience", 3)
        self.lr_schedule = train_cfg.get("lr_schedule", "type1")

        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.result_dir / "checkpoint.pth"

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self) -> nn.Module:
        """
        Run the full training loop.

        Writes metadata.json at start and metrics.json at end.

        Returns:
            The best model (weights loaded from checkpoint).
        """
        self._write_metadata()

        early_stopping = EarlyStopping(patience=self.patience, verbose=True)
        best_val_loss = float("inf")
        best_epoch = 0

        t_start = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._validate()

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

            early_stopping(val_loss, self.model, str(self.checkpoint_path))
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            adjust_learning_rate(
                self.optimizer,
                epoch=epoch,
                initial_lr=self.learning_rate,
                schedule=self.lr_schedule,
                total_epochs=self.epochs,
            )

        training_time = time.time() - t_start

        self.model.load_state_dict(torch.load(str(self.checkpoint_path), weights_only=True))

        run_metrics = {
            "training_time_seconds": round(training_time, 1),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 8),
        }
        self._write_metrics(run_metrics)

        return self.model

    def _train_epoch(self) -> float:
        self.model.train()
        losses: list[float] = []
        t0 = time.time()

        for i, (batch_x, _) in enumerate(self.train_loader):
            batch_x = batch_x.float().to(self.device)
            self.optimizer.zero_grad()
            x_hat = self.model(batch_x)
            loss = self.criterion(x_hat, batch_x)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  iter {i+1}/{len(self.train_loader)} | loss={loss.item():.7f} | {elapsed:.1f}s")

        return float(sum(losses) / len(losses))

    def _validate(self) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_x, _ in self.val_loader:
                batch_x = batch_x.float().to(self.device)
                x_hat = self.model(batch_x)
                loss = self.criterion(x_hat, batch_x)
                losses.append(loss.item())
        self.model.train()
        return float(sum(losses) / len(losses))

    def _write_metadata(self) -> None:
        """
        Write experiment metadata to result_dir/metadata.json.

        Written at run start so the config is captured even if training crashes.
        """
        import flwr

        git_commit = git_branch = None
        git_dirty = None
        try:
            import git as gitpkg
            repo = gitpkg.Repo(search_parent_directories=True)
            git_commit = repo.head.commit.hexsha
            git_branch = repo.active_branch.name
            git_dirty = repo.is_dirty()
        except Exception:
            pass

        meta = {
            "experiment_id": self.result_dir.name,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_dirty": git_dirty,
            "config": self.config,
            "environment": {
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "flwr_version": flwr.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
                "platform": platform.system().lower(),
                "hostname": platform.node(),
            },
        }

        out = self.result_dir / "metadata.json"
        out.write_text(json.dumps(meta, indent=2))
        print(f"Wrote metadata → {out}")

    def _write_metrics(self, metrics: dict) -> None:
        metrics["experiment_id"] = self.result_dir.name
        out = self.result_dir / "metrics.json"
        # Merge with existing metrics.json if evaluation already wrote one
        if out.exists():
            existing = json.loads(out.read_text())
            existing.update(metrics)
            metrics = existing
        out.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote metrics → {out}")
