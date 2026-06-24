"""
In-process N-client FL simulation — no extra devices, no Ray required.

Runs FedAvg manually: each round, all clients receive the global weights,
train locally, return updated weights + sample counts, and the server does
a weighted average. Detection metrics are collected on the final round.

Output is fl_summary.json in the same schema used by real FL runs so all
downstream analysis tools work unchanged.

Usage:
    # 10 MIT-BIH patients, CNN (best edge efficiency)
    python fl/simulate.py --model CNNAutoencoder

    # Pick a specific model with custom patient list
    python fl/simulate.py --model iTransformer --patients mitbih_100 mitbih_106 mitbih_213

    # Override rounds / LR
    python fl/simulate.py --model CNNAutoencoder --rounds 50 --lr 1e-4
"""

from __future__ import annotations
import argparse
import datetime
from datetime import timezone
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.registry import build_dataloaders
from models.registry import ModelRegistry
from fl.client import PhysioAnomalyClient, get_parameters, set_parameters

REPO_ROOT = Path(__file__).resolve().parent.parent

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

# Per-model training defaults — mirrors the tuned FL scripts
_MODEL_TRAIN_DEFAULTS: dict[str, dict] = {
    "iTransformer":   {"rounds": 200, "lr": 1e-4, "lr_min": 1e-5, "lr_schedule": "none",   "local_epochs": 2},
    "PatchTST":       {"rounds": 150, "lr": 1e-4, "lr_min": 1e-5, "lr_schedule": "cosine", "local_epochs": 1},
    "CNNAutoencoder": {"rounds": 100, "lr": 1e-4, "lr_min": 1e-5, "lr_schedule": "cosine", "local_epochs": 1},
    "TimesNet":       {"rounds": 100, "lr": 1e-4, "lr_min": 1e-5, "lr_schedule": "cosine", "local_epochs": 1},
}

# Default 10 MIT-BIH patients — all have 20K+ arrhythmia and 60K+ normal samples.
# Includes real deployment clients (213=Orin, 106=Pi5). Excludes records with
# <5K arrhythmia (103,112,121,230,…) or <30K normal (104,102,207,…).
DEFAULT_PATIENTS = [
    "mitbih_106", "mitbih_119", "mitbih_200", "mitbih_201", "mitbih_203",
    "mitbih_209", "mitbih_213", "mitbih_221", "mitbih_223", "mitbih_228",
]


# ── Config helpers ──────────────────────────────────────────────────────────────

def _build_data_config(patient: str, seq_len: int, batch_size: int) -> dict:
    return {
        "model": {
            "name":     "placeholder",
            "enc_in":   1,
            "c_out":    1,
            "d_model":  64,
            "d_ff":     128,
            "n_heads":  8,
            "e_layers": 2,
            "dropout":  0.1,
        },
        "data": {
            "patient":          patient,
            "sensor":           "ecg",
            "seq_len":          seq_len,
            "step":             seq_len // 2,
            "train_conditions": ["normal"],
            "val_conditions":   ["normal"],
            "test_conditions":  ["arrhythmia"],
            "train_ratio":      0.7,
            "val_ratio":        0.1,
        },
        "training": {
            "batch_size":  batch_size,
            "num_workers": 0,
        },
    }


def _build_model_ns(model_name: str, seq_len: int) -> SimpleNamespace:
    preset = {**_MODEL_PRESETS.get(model_name, {})}
    base = dict(
        seq_len  = seq_len,
        enc_in   = 1,
        c_out    = 1,
        d_model  = preset.get("d_model",  64),
        d_ff     = preset.get("d_ff",    128),
        n_heads  = preset.get("n_heads",   8),
        e_layers = preset.get("e_layers",  2),
        dropout  = preset.get("dropout", 0.1),
    )
    extras = {k: v for k, v in preset.items() if k not in base}
    return SimpleNamespace(**base, **extras)


# ── FedAvg ──────────────────────────────────────────────────────────────────────

def _fedavg(all_params: list[list[np.ndarray]], n_samples: list[int]) -> list[np.ndarray]:
    """Weighted average of parameter arrays by dataset size."""
    weights = np.array(n_samples, dtype=float)
    weights /= weights.sum()
    return [
        sum(w * p[i] for w, p in zip(weights, all_params))
        for i in range(len(all_params[0]))
    ]


def _cosine_lr(rnd: int, rounds: int, lr: float, lr_min: float) -> float:
    progress = (rnd - 1) / max(rounds - 1, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))


# ── Main simulation ─────────────────────────────────────────────────────────────

def run_simulation(
    model_name: str,
    patients:   list[str],
    rounds:     int,
    lr:         float,
    lr_min:     float,
    lr_schedule: str,
    local_epochs: int,
    batch_size: int,
    seq_len:    int,
    device:     torch.device,
) -> None:
    n_clients = len(patients)
    print(f"\nSimulation: {model_name}  |  {n_clients} clients  |  {rounds} rounds")
    print(f"LR: {lr}" + (f" → {lr_min} (cosine)" if lr_schedule == "cosine" else " (flat)"))
    print(f"local_epochs={local_epochs}  batch_size={batch_size}  seq_len={seq_len}")
    print(f"Device: {device}\n")

    # ── Build clients ──────────────────────────────────────────────────────────
    clients: list[PhysioAnomalyClient] = []
    n_train_per_client: list[int] = []
    failed: list[str] = []

    for pid in patients:
        try:
            data_cfg = _build_data_config(pid, seq_len, batch_size)
            model_cfg = {**data_cfg["model"], "name": model_name,
                         **_MODEL_PRESETS.get(model_name, {})}
            data_cfg["model"] = model_cfg
            train_loader, val_loader, test_loader = build_dataloaders(data_cfg)

            ns = _build_model_ns(model_name, seq_len)
            model_cls = ModelRegistry.get(model_name)
            model = model_cls(ns).to(device)

            client = PhysioAnomalyClient(
                model, train_loader, val_loader, device,
                test_loader=test_loader,
                train_label="normal",
                seq_len=seq_len,
                enc_in=1,
            )
            clients.append(client)
            n_train_per_client.append(len(train_loader.dataset))
            print(f"  [{len(clients):2d}] {pid:<18}  train={len(train_loader.dataset):5d}  val={len(val_loader.dataset):4d}")
        except Exception as e:
            print(f"  SKIP {pid}: {e}")
            failed.append(pid)

    if len(clients) < 2:
        raise RuntimeError(f"Need at least 2 clients — only {len(clients)} built successfully")

    n_clients = len(clients)
    print(f"\n{n_clients} clients ready ({len(failed)} skipped)\n")

    # ── Initial global parameters (from client 0's freshly initialized model) ──
    global_params = get_parameters(clients[0].model)

    n_params  = sum(p.numel() for p in clients[0].model.parameters())
    param_mb  = round(n_params * 4 / 1e6, 4)
    param_bytes = n_params * 4
    print(f"Model: {n_params:,} params  {param_mb} MB\n")

    # ── FL rounds ──────────────────────────────────────────────────────────────
    round_history: list[dict] = []
    sim_start = time.time()

    for rnd in range(1, rounds + 1):
        rnd_start = time.time()
        is_final  = (rnd == rounds)

        # Compute LR for this round
        if lr_schedule == "cosine":
            round_lr = _cosine_lr(rnd, rounds, lr, lr_min)
        else:
            round_lr = lr

        fit_config = {
            "local_epochs":  local_epochs,
            "learning_rate": round_lr,
            "round":         rnd,
        }
        eval_config = {"is_final_round": is_final}

        # ── Fit all clients ────────────────────────────────────────────────────
        all_updated_params: list[list[np.ndarray]] = []
        all_n_samples:      list[int]               = []
        client_fit_times:   list[float]             = []

        for client in clients:
            updated, n_samples, fit_metrics = client.fit(global_params, fit_config)
            all_updated_params.append(updated)
            all_n_samples.append(n_samples)
            client_fit_times.append(fit_metrics.get("fit_time_seconds", 0.0))

        # ── FedAvg aggregation ─────────────────────────────────────────────────
        global_params = _fedavg(all_updated_params, all_n_samples)

        # ── Evaluate all clients ───────────────────────────────────────────────
        all_val_losses:   list[float] = []
        all_n_val:        list[int]   = []
        client_eval_times: list[float] = []
        final_metrics:    list[dict]  = []

        for client in clients:
            val_loss, n_val, eval_metrics = client.evaluate(global_params, eval_config)
            all_val_losses.append(val_loss)
            all_n_val.append(n_val)
            client_eval_times.append(eval_metrics.get("eval_time_seconds", 0.0))
            if is_final and "auroc" in eval_metrics:
                final_metrics.append(dict(eval_metrics))

        # Weighted-average val_loss
        total_val = sum(all_n_val)
        weighted_loss = sum(l * n for l, n in zip(all_val_losses, all_n_val)) / total_val

        # Per-round communication: param_bytes broadcast to all clients + received from all
        comm_bytes_out = param_bytes * n_clients
        comm_bytes_in  = param_bytes * n_clients
        comm_total_mb  = round((comm_bytes_out + comm_bytes_in) / 1e6, 4)

        rnd_wall = time.time() - rnd_start
        stat: dict = {
            "round":                    rnd,
            "learning_rate":            round(round_lr, 8),
            "val_loss":                 round(weighted_loss, 8),
            "round_wall_seconds":       round(rnd_wall, 2),
            "comm_bytes_in":            comm_bytes_in,
            "comm_bytes_out":           comm_bytes_out,
            "comm_total_mb":            comm_total_mb,
            "n_clients_fit":            n_clients,
            "client_fit_times_seconds": [round(t, 3) for t in client_fit_times],
            "avg_client_fit_seconds":   round(sum(client_fit_times) / n_clients, 3),
            "client_eval_times_seconds": [round(t, 3) for t in client_eval_times],
            "avg_client_eval_seconds":   round(sum(client_eval_times) / n_clients, 3),
        }
        round_history.append(stat)

        # Progress line — every round but condensed
        print(f"  r{rnd:3d}/{rounds}  loss={weighted_loss:.6f}  "
              f"fit={sum(client_fit_times)/n_clients:.1f}s/client  "
              f"comm={comm_total_mb:.2f}MB  lr={round_lr:.2e}")

    elapsed = time.time() - sim_start

    # ── Save results ───────────────────────────────────────────────────────────
    wall_times   = [e["round_wall_seconds"] for e in round_history]
    comm_mbs     = [e["comm_total_mb"]      for e in round_history]
    all_fit_times = [t for e in round_history for t in e.get("client_fit_times_seconds", [])]
    val_losses   = [e["val_loss"] for e in round_history]
    best_entry   = min(round_history, key=lambda e: e["val_loss"])

    now           = datetime.datetime.now(timezone.utc)
    ts            = now.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{ts}_sim_{model_name}_{n_clients}clients"
    result_dir    = REPO_ROOT / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": experiment_id,
        "timestamp":     now.isoformat(),
        "simulation":    True,
        "patients":      [p for p in patients if p not in failed],
        "patients_skipped": failed,

        "config": {
            "model":    {"name": model_name, **_MODEL_PRESETS.get(model_name, {}),
                         "enc_in": 1, "c_out": 1},
            "data":     {"seq_len": seq_len, "sensor": "ecg",
                         "train_conditions": ["normal"], "test_conditions": ["arrhythmia"]},
            "training": {"batch_size": batch_size},
            "fl":       {"rounds": rounds, "min_clients": n_clients,
                         "local_epochs": local_epochs, "learning_rate": lr,
                         "lr_min": lr_min, "lr_schedule": lr_schedule},
        },

        "fl_run": {
            "rounds_completed": len(round_history),
            "rounds_requested": rounds,
            "n_clients":        n_clients,
            "local_epochs":     local_epochs,
            "learning_rate":    lr,
            "lr_min":           lr_min,
            "lr_schedule":      lr_schedule,
        },

        "model_size": {"n_params": n_params, "param_mb": param_mb},

        "timing": {
            "total_seconds":          round(elapsed, 1),
            "avg_round_seconds":      round(sum(wall_times) / len(wall_times), 2),
            "min_round_seconds":      round(min(wall_times), 2),
            "max_round_seconds":      round(max(wall_times), 2),
            "avg_client_fit_seconds": round(sum(all_fit_times) / len(all_fit_times), 3),
        },

        "communication": {
            "avg_round_mb": round(sum(comm_mbs) / len(comm_mbs), 4),
            "total_mb":     round(sum(comm_mbs), 4),
        },

        "round_history": round_history,

        "final_val_loss": val_losses[-1],
        "best_round":     best_entry["round"],
        "best_val_loss":  best_entry["val_loss"],

        "detection": final_metrics if final_metrics else None,
    }

    out = result_dir / "fl_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # ── Terminal summary ───────────────────────────────────────────────────────
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    W = 70

    print(f"\n{'='*W}")
    print(f"  Simulation Complete  —  {experiment_id}")
    print(f"{'='*W}")
    print(f"  Model      : {model_name}  ({n_params:,} params, {param_mb} MB)")
    print(f"  Clients    : {n_clients}   patients: {', '.join(p.replace('mitbih_','') for p in patients if p not in failed)}")
    print(f"  Rounds     : {len(round_history)}/{rounds}   local_epochs={local_epochs}")
    print(f"  Total time : {mins}m {secs}s")
    print(f"  Avg round  : {sum(wall_times)/len(wall_times):.1f}s")
    print(f"  Comm total : {sum(comm_mbs):.2f} MB  ({sum(comm_mbs)/len(comm_mbs):.2f} MB/round)")
    print(f"  Final loss : {val_losses[-1]:.6f}")
    print(f"  Best loss  : {best_entry['val_loss']:.6f}  (round {best_entry['round']})")

    if final_metrics:
        print(f"\n  Detection metrics (per client):")
        for i, (pid, m) in enumerate(zip([p for p in patients if p not in failed], final_metrics)):
            print(f"    {pid:<18}  AUROC={m.get('auroc','?'):.4f}  "
                  f"F1(PA)={m.get('f1_pa','?'):.4f}  "
                  f"Sep={m.get('score_separation','?'):.2f}σ  "
                  f"CPU={m.get('cpu_latency_ms','?'):.2f}ms")

    print(f"\n  Saved → results/{experiment_id}/fl_summary.json")
    print(f"{'='*W}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-process N-client FL simulation (no Ray, no extra devices)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    default="CNNAutoencoder",
                        choices=list(_MODEL_PRESETS),
                        help="Model architecture to simulate")
    parser.add_argument("--patients", nargs="+", default=None,
                        metavar="PATIENT",
                        help="Patient IDs to simulate (default: 10 MIT-BIH records)")
    parser.add_argument("--rounds",        type=int,   default=None,
                        help="FL rounds (default: per-model tuned value)")
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--lr-min",        type=float, default=None, dest="lr_min")
    parser.add_argument("--lr-schedule",   default=None, choices=["cosine", "none"],
                        dest="lr_schedule")
    parser.add_argument("--local-epochs",  type=int,   default=None, dest="local_epochs")
    parser.add_argument("--batch-size",    type=int,   default=32,   dest="batch_size")
    parser.add_argument("--seq-len",       type=int,   default=128,  dest="seq_len")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--cpu",           action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    # Resolve training config: CLI overrides per-model defaults
    model_defaults = _MODEL_TRAIN_DEFAULTS.get(args.model, {})
    rounds      = args.rounds       or model_defaults.get("rounds",      100)
    lr          = args.lr           or model_defaults.get("lr",          1e-4)
    lr_min      = args.lr_min       if args.lr_min is not None else model_defaults.get("lr_min", 1e-5)
    lr_schedule = args.lr_schedule  or model_defaults.get("lr_schedule", "none")
    local_epochs= args.local_epochs or model_defaults.get("local_epochs", 1)
    patients    = args.patients     or DEFAULT_PATIENTS

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    run_simulation(
        model_name   = args.model,
        patients     = patients,
        rounds       = rounds,
        lr           = lr,
        lr_min       = lr_min,
        lr_schedule  = lr_schedule,
        local_epochs = local_epochs,
        batch_size   = args.batch_size,
        seq_len      = args.seq_len,
        device       = device,
    )


if __name__ == "__main__":
    main()
