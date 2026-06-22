"""
Anomaly detection evaluation.

Supports two evaluation modes:

1. Labeled mode (POC / benchmark):
   Uses condition as a proxy anomaly label. Computes threshold via percentile
   of combined train+test reconstruction energy, then reports:
     - AUROC, AUPRC  (threshold-independent — primary ranking metrics)
     - F1/precision/recall with Point-Adjust protocol (PA — literature standard)
     - F1/precision/recall without Point-Adjust (raw — honest baseline)
     - Score statistics: mean/std for normal and anomaly groups, delta, separation ratio

2. Unlabeled mode (real deployment):
   No ground-truth labels. Reports reconstruction score distributions,
   alert rate, and score delta for live monitoring.

Source reference: tslib/exp/exp_anomaly_detection.py (test method)
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from training.utils import adjustment


class Evaluator:
    """
    Compute anomaly scores and evaluate a trained reconstruction model.

    Args:
        model:         Trained reconstruction model: model(x) → x_hat
        train_loader:  DataLoader for training data (used to fit threshold)
        test_loader:   DataLoader for test data
        anomaly_ratio: Percentile parameter for threshold:
                       threshold = percentile(concat(train_energy, test_energy),
                                             100 - anomaly_ratio)
                       Matches tslib convention. Typical value: 1.0
        device:        torch.device (defaults to CPU)
        result_dir:    If given, evaluation metrics are written to
                       {result_dir}/metrics.json
        train_label:   Condition name treated as normal (label=0)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        anomaly_ratio: float = 1.0,
        device: torch.device | None = None,
        result_dir: str | Path | None = None,
        train_label: str = "baseline",
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.anomaly_ratio = anomaly_ratio
        self.device = device or torch.device("cpu")
        self.result_dir = Path(result_dir) if result_dir else None
        self.train_label = train_label

        self._criterion = nn.MSELoss(reduction="none")

    def compute_scores(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute per-window reconstruction energy and collect condition labels.

        Energy[i] = mean over (seq_len × channels) of MSE(x[i], x_hat[i])

        Source: tslib test() — `score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)`
        Note: tslib uses dim=-1 (mean over channels only, keeping seq_len).
        We additionally mean over seq_len to get one scalar per window,
        matching the concatenate-reshape-(-1) pattern that follows.

        Args:
            loader: DataLoader yielding (window, condition_label) pairs

        Returns:
            scores: float32 array [N_windows] — one energy value per window
            labels: str array [N_windows] — condition label per window
        """
        self.model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_cond in loader:
                batch_x = batch_x.float().to(self.device)
                x_hat = self.model(batch_x)
                # [B, L, C] → mean over channels → [B, L] → mean over time → [B]
                per_channel = self._criterion(batch_x, x_hat)  # [B, L, C]
                score = per_channel.mean(dim=-1).mean(dim=-1)  # [B]
                all_scores.append(score.cpu().numpy())

                if isinstance(batch_cond, (list, tuple)):
                    all_labels.append(np.array(batch_cond))
                else:
                    all_labels.append(np.array([batch_cond]))

        self.model.train()
        scores = np.concatenate(all_scores, axis=0).reshape(-1).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0).reshape(-1)
        return scores, labels

    def compute_threshold(
        self,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
    ) -> float:
        """
        Compute the anomaly detection threshold.

        threshold = percentile(concat(train_scores, test_scores), 100 - anomaly_ratio)

        Source: tslib/exp/exp_anomaly_detection.py line 173
        """
        combined = np.concatenate([train_scores, test_scores], axis=0)
        return float(np.percentile(combined, 100 - self.anomaly_ratio))

    def evaluate_labeled(
        self,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        threshold: float,
        test_labels: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate with ground-truth or proxy labels.

        Computes threshold-independent metrics (AUROC, AUPRC), PA-adjusted
        classification metrics (literature standard), raw classification metrics
        (honest baseline without PA inflation), and score distribution statistics.

        For POC: any condition != train_label is treated as anomalous (label=1).

        Args:
            train_scores: Per-window anomaly scores on normal (training) data [N]
            test_scores:  Per-window anomaly scores on test data [N]
            threshold:    Decision threshold
            test_labels:  Condition string labels [N] — or binary int labels
        """
        # Convert condition strings to binary labels
        if test_labels.dtype.kind in ("U", "S", "O"):
            gt = (test_labels != self.train_label).astype(int)
        else:
            gt = test_labels.astype(int)

        # ── Threshold-independent metrics ──────────────────────────────────────
        # WESAD and similar datasets have a test set that is 100% anomalous
        # (stress-only), so sklearn can't compute AUROC from test labels alone.
        # Solution: use train_scores as the normal class reference and concatenate
        # with test_scores. This is the correct framing for reconstruction-based AD:
        # "how well does reconstruction error separate normal windows from anomalous ones?"
        n_pos = gt.sum()
        n_neg = len(gt) - n_pos
        if n_pos > 0:
            auroc_scores = np.concatenate([train_scores, test_scores])
            auroc_labels = np.concatenate([
                np.zeros(len(train_scores), dtype=int),
                gt,  # preserves mixed test sets; all-anomaly test sets work too
            ])
            auroc = float(roc_auc_score(auroc_labels, auroc_scores))
            auprc = float(average_precision_score(auroc_labels, auroc_scores))
        else:
            auroc = auprc = float("nan")

        # ── Raw classification (no Point-Adjust) ──────────────────────────────
        pred_raw = (test_scores > threshold).astype(int)
        prec_r, rec_r, f1_r, _ = precision_recall_fscore_support(
            gt, pred_raw, average="binary", zero_division=0,
        )

        # ── PA-adjusted classification (literature standard) ──────────────────
        pred_pa = (test_scores > threshold).astype(int)
        gt_adj, pred_pa = adjustment(gt.copy(), pred_pa)
        prec_pa, rec_pa, f1_pa, _ = precision_recall_fscore_support(
            gt_adj, pred_pa, average="binary", zero_division=0,
        )

        # ── Score distribution statistics ─────────────────────────────────────
        anomaly_scores = test_scores[gt == 1] if n_pos > 0 else np.array([float("nan")])
        normal_scores  = train_scores  # train set is purely normal by construction

        mean_normal   = float(np.mean(normal_scores))
        std_normal    = float(np.std(normal_scores))
        mean_anomaly  = float(np.mean(anomaly_scores))
        score_delta   = mean_anomaly - mean_normal
        # How many normal-score standard deviations separate the two distributions
        score_sep     = score_delta / std_normal if std_normal > 0 else float("nan")

        return {
            "threshold":      round(threshold, 8),
            # Threshold-independent (primary comparison metrics)
            "auroc":          round(auroc, 6),
            "auprc":          round(auprc, 6),
            # PA-adjusted (matches published literature)
            "f1_pa":          round(float(f1_pa), 6),
            "precision_pa":   round(float(prec_pa), 6),
            "recall_pa":      round(float(rec_pa), 6),
            # Raw (honest, no PA inflation)
            "f1_raw":         round(float(f1_r), 6),
            "precision_raw":  round(float(prec_r), 6),
            "recall_raw":     round(float(rec_r), 6),
            # Score distributions
            "mean_normal_score":  round(mean_normal, 8),
            "std_normal_score":   round(std_normal, 8),
            "mean_anomaly_score": round(mean_anomaly, 8),
            "score_delta":        round(score_delta, 8),
            "score_separation":   round(score_sep, 4),
        }

    def evaluate_unlabeled(
        self,
        train_scores: np.ndarray,
        test_scores: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        """
        Evaluate without ground-truth labels (real deployment mode).

        Args:
            train_scores: Reconstruction energy on training (normal) data
            test_scores:  Reconstruction energy on deployment data
            threshold:    Decision threshold

        Returns:
            Dict with: alert_rate, mean_train_score, mean_test_score,
                       score_delta, threshold
        """
        alert_rate = float(np.mean(test_scores > threshold))
        return {
            "threshold": round(threshold, 8),
            "alert_rate": round(alert_rate, 6),
            "mean_train_score": round(float(np.mean(train_scores)), 8),
            "mean_test_score": round(float(np.mean(test_scores)), 8),
            "score_delta": round(float(np.mean(test_scores) - np.mean(train_scores)), 8),
        }

    def run(self, labeled: bool = True) -> dict:
        """
        Run the full evaluation pipeline.

        Args:
            labeled: If True, use labeled mode (PA protocol + classification metrics).
                     If False, use unlabeled mode (score distribution + alert rate).

        Returns:
            Metrics dict. Written to result_dir/metrics.json if result_dir is set.
        """
        print("Computing train scores…")
        train_scores, train_labels = self.compute_scores(self.train_loader)

        print("Computing test scores…")
        test_scores, test_labels = self.compute_scores(self.test_loader)

        threshold = self.compute_threshold(train_scores, test_scores)
        print(f"Threshold: {threshold:.6f}")

        n_train = len(train_scores)
        n_test = len(test_scores)

        if labeled:
            metrics = self.evaluate_labeled(train_scores, test_scores, threshold, test_labels)
            metrics["evaluation_mode"] = "labeled"
        else:
            metrics = self.evaluate_unlabeled(train_scores, test_scores, threshold)
            metrics["evaluation_mode"] = "unlabeled"

        metrics["n_train_windows"] = n_train
        metrics["n_test_windows"] = n_test

        print(
            "AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
            "F1(PA)={f1_pa:.4f}  F1(raw)={f1_raw:.4f}  "
            "Score sep={score_separation:.2f}σ".format(**metrics)
            if labeled
            else "Alert rate={alert_rate:.4f}  Score delta={score_delta:.6f}".format(**metrics)
        )

        if self.result_dir:
            out = self.result_dir / "metrics.json"
            if out.exists():
                existing = json.loads(out.read_text())
                existing.update(metrics)
                metrics = existing
            out.write_text(json.dumps(metrics, indent=2))
            print(f"Wrote metrics → {out}")

        return metrics
