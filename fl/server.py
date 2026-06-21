"""
Federated Learning aggregation server.

Runs a Flower FedAvg server that coordinates model weight aggregation
across multiple edge clients. The server has no local data and no
knowledge of the model architecture — it only aggregates parameter arrays.

After all rounds complete, writes results/{id}/fl_summary.json with:
  - Per-round val_loss, wall time, communication overhead, client compute times
  - Model size (params, MB)
  - Aggregate timing and communication totals
  - Full run config

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
from flwr.common import parameters_to_ndarrays

REPO_ROOT = Path(__file__).resolve().parent.parent


def _params_bytes(parameters) -> int:
    """Size in bytes of a Flower Parameters object (serialized tensors)."""
    try:
        return sum(len(t) for t in parameters.tensors)
    except Exception:
        return 0


class TimedFedAvg(FedAvg):
    """
    FedAvg extended with per-round timing and communication tracking.

    Records for each round:
        round_wall_seconds   — server-side wall time (client compute + network RTT)
        comm_bytes_in        — bytes received from all clients (updated weights)
        comm_bytes_out       — bytes sent to all clients (global weights broadcast)
        comm_total_mb        — total bidirectional communication in MB
        client_fit_times     — list of per-client local training times (seconds)
        client_eval_times    — list of per-client evaluation times (seconds)
        n_params / param_mb  — model size (from first client response, round 1 only)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._round_start: dict[int, float] = {}
        self._bytes_out:   dict[int, int]   = {}
        self.round_stats:  list[dict]        = []
        self._model_size:  dict | None       = None

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start[server_round] = time.time()
        # Record bytes being broadcast to each client
        per_client_bytes = _params_bytes(parameters)
        n_clients = len(client_manager.all())
        self._bytes_out[server_round] = per_client_bytes * max(n_clients, 1)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        wall = time.time() - self._round_start.get(server_round, time.time())

        bytes_in = sum(_params_bytes(fit_res.parameters) for _, fit_res in results)
        bytes_out = self._bytes_out.get(server_round, 0)
        total_mb = round((bytes_in + bytes_out) / 1e6, 4)

        client_fit_times = [
            fit_res.metrics["fit_time_seconds"]
            for _, fit_res in results
            if "fit_time_seconds" in fit_res.metrics
        ]

        # Capture model size once from the first round's first client
        if self._model_size is None and results:
            _, first = results[0]
            if "n_params" in first.metrics:
                self._model_size = {
                    "n_params":  int(first.metrics["n_params"]),
                    "param_mb":  float(first.metrics.get("param_mb", 0)),
                }

        stat: dict = {
            "round":               server_round,
            "round_wall_seconds":  round(wall, 2),
            "comm_bytes_in":       bytes_in,
            "comm_bytes_out":      bytes_out,
            "comm_total_mb":       total_mb,
            "n_clients_fit":       len(results),
        }
        if client_fit_times:
            stat["client_fit_times_seconds"] = [round(t, 3) for t in client_fit_times]
            stat["avg_client_fit_seconds"]   = round(sum(client_fit_times) / len(client_fit_times), 3)

        self.round_stats.append(stat)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        # Attach eval times to the matching round stat
        eval_times = [
            fit_res.metrics["eval_time_seconds"]
            for _, fit_res in results
            if "eval_time_seconds" in fit_res.metrics
        ]
        for stat in self.round_stats:
            if stat["round"] == server_round and eval_times:
                stat["client_eval_times_seconds"] = [round(t, 3) for t in eval_times]
                stat["avg_client_eval_seconds"]   = round(sum(eval_times) / len(eval_times), 3)
        return super().aggregate_evaluate(server_round, results, failures)


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Weighted average of val_loss across all reporting clients."""
    losses = [num * m["val_loss"] for num, m in metrics]
    total  = sum(num for num, _ in metrics)
    return {"val_loss": sum(losses) / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="FLIoMT FL Server")
    parser.add_argument("--config",        type=str,   default=None)
    parser.add_argument("--host",          type=str,   default=None)
    parser.add_argument("--port",          type=int,   default=None)
    parser.add_argument("--rounds",        type=int,   default=None)
    parser.add_argument("--min_clients",   type=int,   default=None)
    parser.add_argument("--local_epochs",  type=int,   default=None)
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

    strategy = TimedFedAvg(
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

    _save_results(history, strategy, cfg, rounds, min_clients, local_epochs, lr, elapsed)


def _save_results(
    history,
    strategy: TimedFedAvg,
    cfg: dict,
    rounds: int,
    min_clients: int,
    local_epochs: int,
    lr: float,
    elapsed: float,
) -> None:
    # Per-round val_loss from Flower History
    round_val_loss: dict[int, float] = {}
    if hasattr(history, "metrics_distributed"):
        for rnd, val in history.metrics_distributed.get("val_loss", []):
            round_val_loss[int(rnd)] = float(val)
    if not round_val_loss and hasattr(history, "losses_distributed"):
        for rnd, loss in history.losses_distributed:
            round_val_loss[int(rnd)] = float(loss)

    # Merge val_loss into per-round timing stats
    round_history = []
    for stat in strategy.round_stats:
        rnd = stat["round"]
        entry = dict(stat)
        if rnd in round_val_loss:
            entry["val_loss"] = round(round_val_loss[rnd], 8)
        round_history.append(entry)

    # Aggregate summaries
    val_losses = [e["val_loss"] for e in round_history if "val_loss" in e]
    wall_times = [e["round_wall_seconds"] for e in round_history]
    comm_mbs   = [e["comm_total_mb"] for e in round_history]
    all_fit_times = [
        t for e in round_history
        for t in e.get("client_fit_times_seconds", [])
    ]

    final_val_loss = val_losses[-1] if val_losses else None
    best_entry     = min(round_history, key=lambda e: e.get("val_loss", float("inf"))) if round_history else {}

    ts            = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_name    = cfg.get("model", {}).get("name", "unknown")
    experiment_id = cfg.get("experiment", {}).get("name", f"fl_{model_name}")
    experiment_id = f"{ts}_{experiment_id}"

    result_dir = REPO_ROOT / "results" / experiment_id
    result_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": experiment_id,
        "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "config":        cfg,

        "fl_run": {
            "rounds_completed": len(round_history),
            "rounds_requested": rounds,
            "min_clients":      min_clients,
            "local_epochs":     local_epochs,
            "learning_rate":    lr,
        },

        "model_size": strategy._model_size,

        "timing": {
            "total_seconds":     round(elapsed, 1),
            "avg_round_seconds": round(sum(wall_times) / len(wall_times), 2) if wall_times else None,
            "min_round_seconds": round(min(wall_times), 2) if wall_times else None,
            "max_round_seconds": round(max(wall_times), 2) if wall_times else None,
            "avg_client_fit_seconds": round(sum(all_fit_times) / len(all_fit_times), 3) if all_fit_times else None,
        },

        "communication": {
            "avg_round_mb":  round(sum(comm_mbs) / len(comm_mbs), 4) if comm_mbs else None,
            "total_mb":      round(sum(comm_mbs), 4),
        },

        "round_history": round_history,

        "final_val_loss": final_val_loss,
        "best_round":     best_entry.get("round"),
        "best_val_loss":  best_entry.get("val_loss"),
    }

    out = result_dir / "fl_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # Terminal summary
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    ms   = strategy._model_size

    print("\n" + "=" * 60)
    print(f"  FL Run Complete  —  {experiment_id}")
    print("=" * 60)
    print(f"  Model        : {model_name}" + (
        f"  ({ms['n_params']:,} params, {ms['param_mb']} MB)" if ms else ""
    ))
    print(f"  Rounds       : {len(round_history)}/{rounds}")
    print(f"  Clients      : {min_clients}   local_epochs={local_epochs}   lr={lr}")
    print(f"  Total time   : {mins}m {secs}s")
    if wall_times:
        print(f"  Round time   : avg={sum(wall_times)/len(wall_times):.1f}s  "
              f"min={min(wall_times):.1f}s  max={max(wall_times):.1f}s")
    if all_fit_times:
        print(f"  Client train : avg={sum(all_fit_times)/len(all_fit_times):.1f}s per client per round")
    if comm_mbs:
        print(f"  Comm (total) : {sum(comm_mbs):.2f} MB  ({sum(comm_mbs)/len(comm_mbs):.2f} MB/round)")
    if final_val_loss is not None:
        print(f"  Final loss   : {final_val_loss:.6f}")
    if best_entry:
        print(f"  Best loss    : {best_entry.get('val_loss', '?'):.6f}  (round {best_entry.get('round', '?')})")

    if round_history:
        print(f"\n  Round  val_loss    wall(s)  comm(MB)  avg_client_fit(s)")
        print(f"  {'─'*55}")
        for e in round_history:
            fit_s = f"{e['avg_client_fit_seconds']:.1f}" if "avg_client_fit_seconds" in e else "  — "
            vl    = f"{e['val_loss']:.6f}" if "val_loss" in e else "  —    "
            print(f"  {e['round']:3d}    {vl}   {e['round_wall_seconds']:6.1f}   "
                  f"{e['comm_total_mb']:6.3f}    {fit_s}")

    print(f"\n  Saved → results/{experiment_id}/fl_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
