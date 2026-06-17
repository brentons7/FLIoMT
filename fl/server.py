"""
Federated Learning aggregation server.

Runs a Flower FedAvg server that coordinates model weight aggregation
across multiple edge clients. The server has no local data and no
knowledge of the model architecture — it only aggregates parameter arrays.

After all rounds complete, writes a results/fl_summary.json with per-round
val_loss history, run config, timing, and prints a human-readable summary.

Usage:
    python fl/server.py --config configs/experiments/fl_ecg_3client.yaml

Or via shell script:
    bash scripts/fl/start_server.sh
"""

from __future__ import annotations
import argparse
import datetime
import json
import time
from pathlib import Path

import yaml
import flwr as fl
from flwr.server.strategy import FedAvg

REPO_ROOT = Path(__file__).resolve().parent.parent


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Weighted average of val_loss across all reporting clients."""
    losses = [num * m["val_loss"] for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"val_loss": sum(losses) / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="FLIoMT FL Server")
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment YAML config")
    parser.add_argument("--host", type=str, default=None,
                        help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--min_clients", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    cfg: dict = {}
    fl_cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        fl_cfg = cfg.get("fl", {})

    host         = args.host          or "0.0.0.0"
    port         = args.port          or fl_cfg.get("port", 8080)
    rounds       = args.rounds        or fl_cfg.get("rounds", 10)
    min_clients  = args.min_clients   or fl_cfg.get("min_clients", 2)
    local_epochs = args.local_epochs  or fl_cfg.get("local_epochs", 1)
    lr           = args.learning_rate or fl_cfg.get("learning_rate", 1e-4)

    def fit_config(server_round: int) -> dict:
        return {"local_epochs": local_epochs, "learning_rate": lr, "round": server_round}

    strategy = FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting FL server on {host}:{port}")
    print(f"  rounds={rounds}  min_clients={min_clients}  local_epochs={local_epochs}  lr={lr}")

    t_start = time.time()
    history = fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
    elapsed = time.time() - t_start

    _save_results(history, cfg, rounds, min_clients, local_epochs, lr, elapsed)


def _save_results(
    history,
    cfg: dict,
    rounds: int,
    min_clients: int,
    local_epochs: int,
    lr: float,
    elapsed: float,
) -> None:
    # Extract per-round val_loss from Flower's History object.
    # evaluate_metrics_aggregation_fn output lands in history.metrics_distributed.
    round_losses: dict[int, float] = {}
    if hasattr(history, "metrics_distributed"):
        val_loss_series = history.metrics_distributed.get("val_loss", [])
        for rnd, val in val_loss_series:
            round_losses[int(rnd)] = float(val)

    # Fall back to losses_distributed if metrics weren't aggregated
    if not round_losses and hasattr(history, "losses_distributed"):
        for rnd, loss in history.losses_distributed:
            round_losses[int(rnd)] = float(loss)

    round_history = [
        {"round": r, "val_loss": round(round_losses[r], 8)}
        for r in sorted(round_losses)
    ]

    final_val_loss = round_history[-1]["val_loss"] if round_history else None
    best = min(round_history, key=lambda x: x["val_loss"]) if round_history else {}

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.get("model", {}).get("name", "unknown")
    sensor     = cfg.get("data",  {}).get("sensor",  "unknown")
    patient    = cfg.get("data",  {}).get("patient", "unknown")
    experiment_id = f"{ts}_fl_{model_name}_{sensor}_{patient}"

    result_dir = REPO_ROOT / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": experiment_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config": cfg,
        "fl_run": {
            "rounds_completed": len(round_history),
            "rounds_requested": rounds,
            "min_clients": min_clients,
            "local_epochs": local_epochs,
            "learning_rate": lr,
        },
        "round_history": round_history,
        "final_val_loss": final_val_loss,
        "best_round": best.get("round"),
        "best_val_loss": best.get("val_loss"),
        "total_time_seconds": round(elapsed, 1),
    }

    out = result_dir / "fl_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # Terminal summary
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print("\n" + "=" * 56)
    print(f"  FL Run Complete")
    print(f"  {experiment_id}")
    print("=" * 56)
    print(f"  Model          : {model_name}")
    print(f"  Rounds         : {len(round_history)}/{rounds}")
    print(f"  Clients        : {min_clients}")
    print(f"  Local epochs   : {local_epochs}   LR: {lr}")
    print(f"  Total time     : {mins}m {secs}s")
    if final_val_loss is not None:
        print(f"  Final val loss : {final_val_loss:.6f}")
    if best:
        print(f"  Best val loss  : {best['val_loss']:.6f}  (round {best['round']})")
    if round_history:
        print(f"\n  Round history (val_loss):")
        worst = max(r["val_loss"] for r in round_history)
        for entry in round_history:
            filled = int(30 * entry["val_loss"] / worst) if worst > 0 else 0
            bar = "█" * filled + "░" * (30 - filled)
            print(f"    {entry['round']:3d}  {bar}  {entry['val_loss']:.6f}")
    print(f"\n  Saved → results/{experiment_id}/fl_summary.json")
    print("=" * 56)


if __name__ == "__main__":
    main()
