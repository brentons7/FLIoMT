"""
Federated learning stack using Flower (flwr).

Architecture:
    Server — FedAvg aggregation server; network-only, no local data
    Client — PhysioAnomalyClient wrapping a model and local DataLoaders
    Partition — Strategies: temporal, patient-based, condition-based

Source: tslib/fl/
Migration status: PENDING — FL stack to be ported in Phase 5.
"""
