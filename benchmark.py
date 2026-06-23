#!/usr/bin/env python3
"""
Centralized model benchmark — no FL required.

Trains and evaluates each model on the same data and task used in FL
experiments (reconstruction-based anomaly detection, WESAD ECG by default).
Use this to quickly narrow down which models are worth deploying in FL before
going through the full 3-device round-trip.

Results are printed as a ranked table and saved to results/benchmarks/.

Usage
-----
    # All models, default settings (wesad_S2, 30 epochs)
    python benchmark.py

    # Specific models
    python benchmark.py --models PatchTST CNNAutoencoder TimesNet iTransformer

    # Different patient or dataset
    python benchmark.py --patient wesad_S3
    python benchmark.py --patient mitbih_100 --test-conditions abnormal

    # Tune training
    python benchmark.py --epochs 50 --lr 5e-4 --batch-size 64
"""

from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch import optim

from datasets.registry import build_dataloaders
from models.registry import ModelRegistry
from training.evaluator import Evaluator
from training.utils import EarlyStopping, adjust_learning_rate, measure_edge

# ── Model presets (keep in sync with fl/run_client.py _MODEL_PRESETS) ──────────
_MODEL_PRESETS: dict[str, dict] = {
    "iTransformer": {
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 3, "dropout": 0.1,
    },
    "PatchTST": {
        "d_model": 128, "d_ff": 256, "n_heads": 8, "e_layers": 4,
        "dropout": 0.1, "patch_len": 16, "stride": 8,
    },
    "TimesNet": {
        "d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 2,
        "dropout": 0.1, "top_k": 5, "num_kernels": 6,
    },
    "CNNAutoencoder": {
        "d_model": 32, "d_ff": 64, "n_heads": 1, "e_layers": 6, "dropout": 0.1,
    },
}
_DEFAULT_PRESET: dict = {
    "d_model": 64, "d_ff": 128, "n_heads": 8, "e_layers": 2, "dropout": 0.1,
}

# Models run by --all, in a sensible order (fastest first so you see results quickly)
ALL_MODELS = [
    "CNNAutoencoder",
    "iTransformer",
    "TimesNet",
    "PatchTST",
]


def _build_config(args: argparse.Namespace, model_name: str) -> dict:
    preset = {**_DEFAULT_PRESET, **_MODEL_PRESETS.get(model_name, {})}
    return {
        "model": {
            "name": model_name,
            "enc_in": args.enc_in,
            "c_out":  args.enc_in,
            **preset,
        },
        "data": {
            "patient":          args.patient,
            "sensor":           args.sensor,
            "seq_len":          args.seq_len,
            "step":             args.seq_len // 2,
            "train_conditions": args.train_conditions,
            "val_conditions":   args.train_conditions,
            "test_conditions":  args.test_conditions,
            "train_ratio":      0.7,
            "val_ratio":        0.1,
        },
        "training": {
            "batch_size":  args.batch_size,
            "num_workers": 0,
        },
    }


def _build_model_ns(config: dict) -> SimpleNamespace:
    m = config["model"]
    d = config["data"]
    base = dict(
        seq_len  = d["seq_len"],
        enc_in   = m["enc_in"],
        c_out    = m["c_out"],
        d_model  = m.get("d_model",  64),
        d_ff     = m.get("d_ff",    128),
        n_heads  = m.get("n_heads",   8),
        e_layers = m.get("e_layers",  2),
        dropout  = m.get("dropout", 0.1),
    )
    extras = {k: v for k, v in m.items() if k not in ("name", *base)}
    return SimpleNamespace(**base, **extras)


def _train(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    lr_min: float,
    patience: int,
) -> tuple[float, float]:
    """Train model, return (best_val_loss, elapsed_seconds)."""
    criterion    = nn.MSELoss()
    optimizer    = optim.Adam(model.parameters(), lr=lr)
    early_stop   = EarlyStopping(patience=patience, verbose=False)
    ckpt         = Path("/tmp/_bench_ckpt.pth")
    best_val     = float("inf")
    t0           = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.float().to(device)
            optimizer.zero_grad()
            out  = model(batch_x)
            loss = criterion(out, batch_x)
            assoc = getattr(model, "assoc_loss", None)
            if assoc is not None:
                loss = loss + assoc
            loss.backward()
            optimizer.step()

        # ── validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.float().to(device)
                out = model(batch_x)
                val_losses.append(criterion(out, batch_x).item())
        val_loss = sum(val_losses) / len(val_losses)

        if val_loss < best_val:
            best_val = val_loss

        early_stop(val_loss, model, str(ckpt))
        if early_stop.early_stop:
            break

        adjust_learning_rate(
            optimizer, epoch=epoch, initial_lr=lr,
            schedule="cosine", total_epochs=epochs,
        )

    model.load_state_dict(torch.load(str(ckpt), weights_only=True))
    return best_val, time.time() - t0



def _run_one(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    config = _build_config(args, model_name)

    train_loader, val_loader, test_loader = build_dataloaders(config)

    ns        = _build_model_ns(config)
    model_cls = ModelRegistry.get(model_name)
    model     = model_cls(ns).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    param_mb  = round(n_params * 4 / 1e6, 4)

    print(f"\n{'─'*60}")
    print(f"  {model_name}  ({n_params:,} params  {param_mb} MB)")
    print(f"{'─'*60}")

    best_val, train_secs = _train(
        model, train_loader, val_loader, device,
        epochs=args.epochs, lr=args.lr, lr_min=args.lr_min,
        patience=args.patience,
    )

    evaluator = Evaluator(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        anomaly_ratio=args.anomaly_ratio,
        device=device,
        train_label=args.train_conditions[0],
    )
    metrics = evaluator.run(labeled=True)

    edge = measure_edge(model, args.seq_len, args.enc_in)

    return {
        "model":               model_name,
        "n_params":            n_params,
        "param_mb":            param_mb,
        "best_val_loss":       round(best_val, 8),
        # Task metrics
        "auroc":               metrics["auroc"],
        "auprc":               metrics["auprc"],
        "f1_pa":               metrics["f1_pa"],
        "f1_raw":              metrics["f1_raw"],
        "precision_pa":        metrics["precision_pa"],
        "recall_pa":           metrics["recall_pa"],
        "precision_raw":       metrics["precision_raw"],
        "recall_raw":          metrics["recall_raw"],
        # Score distributions
        "mean_normal_score":   metrics["mean_normal_score"],
        "std_normal_score":    metrics["std_normal_score"],
        "mean_anomaly_score":  metrics["mean_anomaly_score"],
        "score_delta":         metrics["score_delta"],
        "score_separation":    metrics["score_separation"],
        # Edge metrics
        "cpu_latency_ms":      edge["cpu_latency_ms"],
        "gpu_throughput_wps":  edge.get("gpu_throughput_wps"),
        # Training
        "train_secs":          round(train_secs, 1),
    }


def _print_table(results: list[dict]) -> None:
    ok  = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    ok.sort(key=lambda r: r["auroc"], reverse=True)

    W = 108

    # ── Detection quality ─────────────────────────────────────────────────────
    print(f"\n\n{'═'*W}")
    print("  DETECTION QUALITY  (ranked by AUROC — threshold-independent)")
    print(f"{'═'*W}")
    print(f"{'Model':<26} {'AUROC':>7}  {'AUPRC':>7}  {'F1(PA)':>8}  {'F1(raw)':>8}  {'Score Δ':>9}  {'Sep(σ)':>7}  {'Val Loss':>10}")
    print(f"{'─'*W}")
    for r in ok:
        sep = r['score_separation']
        sep_s = f"{sep:>7.2f}" if sep == sep else "     nan"  # nan check
        print(
            f"{r['model']:<26} {r['auroc']:>7.4f}  {r['auprc']:>7.4f}  "
            f"{r['f1_pa']:>8.4f}  {r['f1_raw']:>8.4f}  "
            f"{r['score_delta']:>9.5f}  {sep_s}  {r['best_val_loss']:>10.6f}"
        )

    # ── Edge compatibility ────────────────────────────────────────────────────
    ok_edge = sorted(ok, key=lambda r: r["cpu_latency_ms"])
    print(f"\n{'═'*W}")
    print("  EDGE COMPATIBILITY  (ranked by Pi CPU latency — lower is better)")
    print(f"{'═'*W}")
    gpu_col = torch.cuda.is_available()
    if gpu_col:
        print(f"{'Model':<26} {'Params':>9}  {'Size(MB)':>8}  {'CPU ms/win':>11}  {'GPU wins/s':>11}  {'Train(s)':>9}")
    else:
        print(f"{'Model':<26} {'Params':>9}  {'Size(MB)':>8}  {'CPU ms/win':>11}  {'Train(s)':>9}")
    print(f"{'─'*W}")
    for r in ok_edge:
        if gpu_col:
            gput = r.get("gpu_throughput_wps")
            gput_s = f"{gput:>11,}" if gput is not None else "        n/a"
            print(
                f"{r['model']:<26} {r['n_params']:>9,}  {r['param_mb']:>8.4f}  "
                f"{r['cpu_latency_ms']:>11.3f}  {gput_s}  {r['train_secs']:>9.0f}s"
            )
        else:
            print(
                f"{r['model']:<26} {r['n_params']:>9,}  {r['param_mb']:>8.4f}  "
                f"{r['cpu_latency_ms']:>11.3f}  {r['train_secs']:>9.0f}s"
            )

    if bad:
        print(f"\n{'─'*W}")
        print("  FAILED")
        for r in bad:
            print(f"  {r['model']:<26}  {r['error']}")

    print(f"{'═'*W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FLIoMT models centrally (no FL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Which models ──────────────────────────────────────────────────────────
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="Models to benchmark. Omit to run all.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--patient",          default="mitbih_213")
    parser.add_argument("--sensor",           default="ecg")
    parser.add_argument("--seq-len",          type=int, default=128, dest="seq_len")
    parser.add_argument("--enc-in",           type=int, default=1,   dest="enc_in")
    parser.add_argument("--train-conditions", nargs="+", default=["normal"],
                        dest="train_conditions")
    parser.add_argument("--test-conditions",  nargs="+", default=["arrhythmia"],
                        dest="test_conditions")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--lr-min",       type=float, default=1e-5, dest="lr_min")
    parser.add_argument("--batch-size",   type=int,   default=32,   dest="batch_size")
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--anomaly-ratio",type=float, default=1.0,  dest="anomaly_ratio")

    # ── Hardware ──────────────────────────────────────────────────────────────
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU is available")

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"Patient: {args.patient}  Sensor: {args.sensor}  Seq len: {args.seq_len}")
    print(f"Train on: {args.train_conditions}  Test on: {args.test_conditions}")
    print(f"Epochs: {args.epochs}  LR: {args.lr}→{args.lr_min}  Batch: {args.batch_size}")

    models_to_run = args.models or ALL_MODELS

    results   = []
    run_start = datetime.now(timezone.utc)

    for model_name in models_to_run:
        try:
            r = _run_one(model_name, args, device)
            results.append(r)
        except Exception as e:
            print(f"\n  {model_name} FAILED: {e}")
            results.append({"model": model_name, "error": str(e)})

    _print_table(results)

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts      = run_start.strftime("%Y%m%d_%H%M%S")
    out     = out_dir / f"benchmark_{ts}_{args.patient}.json"

    payload = {
        "timestamp":        run_start.isoformat(),
        "patient":          args.patient,
        "sensor":           args.sensor,
        "seq_len":          args.seq_len,
        "train_conditions": args.train_conditions,
        "test_conditions":  args.test_conditions,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "batch_size":       args.batch_size,
        "results":          results,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"Results saved → {out}\n")


if __name__ == "__main__":
    main()
