"""
Federated Learning aggregation server.

Runs a Flower FedAvg server that coordinates model weight aggregation
across multiple edge clients. The server has no local data and no
knowledge of the model architecture — it only aggregates parameter
arrays.

Source reference: tslib/fl/server.py

Usage:
    python fl/server.py --config configs/experiments/fl_ecg_3client.yaml

Or via shell script:
    bash scripts/fl/start_server.sh
"""

from __future__ import annotations
import argparse

import yaml
import flwr as fl
from flwr.server.strategy import FedAvg


def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Weighted average of val_loss across all reporting clients."""
    losses = [num * m["val_loss"] for num, m in metrics]
    total = sum(num for num, _ in metrics)
    return {"val_loss": sum(losses) / total}


def main() -> None:
    parser = argparse.ArgumentParser(description="FLIoMT FL Server")
    parser.add_argument("--config", type=str, default=None,
                        help="Experiment YAML config (fl section used for defaults)")
    parser.add_argument("--host", type=str, default=None,
                        help="Interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None,
                        help="Number of federated rounds")
    parser.add_argument("--min_clients", type=int, default=None,
                        help="Minimum clients required to start a round")
    parser.add_argument("--local_epochs", type=int, default=None,
                        help="Local training epochs communicated to each client")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate communicated to each client")
    args = parser.parse_args()

    fl_cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        fl_cfg = cfg.get("fl", {})

    host         = args.host         or "0.0.0.0"
    port         = args.port         or fl_cfg.get("port", 8080)
    rounds       = args.rounds       or fl_cfg.get("rounds", 10)
    min_clients  = args.min_clients  or fl_cfg.get("min_clients", 2)
    local_epochs = args.local_epochs or fl_cfg.get("local_epochs", 1)
    lr           = args.learning_rate or fl_cfg.get("learning_rate", 1e-4)

    def fit_config(server_round: int) -> dict:
        return {
            "local_epochs":  local_epochs,
            "learning_rate": lr,
            "round":         server_round,
        }

    strategy = FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print(f"Starting FL server on {host}:{port}")
    print(f"  rounds={rounds}  min_clients={min_clients}  local_epochs={local_epochs}  lr={lr}")

    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
