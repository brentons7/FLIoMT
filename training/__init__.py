"""
Centralized training infrastructure for reconstruction-based anomaly detection.

Modules:
    trainer   — Trainer: training loop, checkpoint management, metadata logging
    evaluator — Evaluator: anomaly threshold, PA protocol, labeled/unlabeled metrics
    utils     — EarlyStopping, learning rate schedules, adjustment(), metrics
"""
