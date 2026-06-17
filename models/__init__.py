"""
Anomaly detection model implementations and registry.

All models follow the reconstruction-based autoencoder paradigm:
    model(x: Tensor[B, L, C]) -> x_hat: Tensor[B, L, C]

The mean squared error MSE(x, x_hat) serves as the per-timestep anomaly score.
No labels are used during training (fully unsupervised).

Modules:
    registry — ModelRegistry with lazy loading and capability declarations
    layers/  — Shared layer implementations (Embedding, Attention, etc.)

Source: tslib/models/ and tslib/layers/
Migration status: PENDING — models to be ported in Phase 3.
"""
