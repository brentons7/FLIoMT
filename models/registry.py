"""
Model registry with lazy loading.

All models implement: model(x: Tensor[B, L, C]) -> x_hat: Tensor[B, L, C]

Usage:
    from models.registry import ModelRegistry

    model_cls = ModelRegistry.get("PatchTST")
    model = model_cls(config)
"""

from __future__ import annotations
import importlib
from dataclasses import dataclass


@dataclass
class ModelEntry:
    module:          str        # Import path
    class_name:      str        # Class name within module
    edge_compatible: bool       # Runs on Pi 5 CPU without CUDA
    layer_deps:      list[str]  # Required files from models/layers/
    notes:           str = ""


_REGISTRY: dict[str, ModelEntry] = {
    "PatchTST": ModelEntry(
        module="models.PatchTST",
        class_name="PatchTST",
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        notes="Best arrhythmia detection (AUROC 0.988 on MIT-BIH). Patch-as-token captures QRS morphology.",
    ),
    "CNNAutoencoder": ModelEntry(
        module="models.CNNAutoencoder",
        class_name="CNNAutoencoder",
        edge_compatible=True,
        layer_deps=[],
        notes=(
            "Fastest on Pi 5 CPU (0.3ms/win, 0.05MB). Dilated residual Conv1d blocks. "
            "RF at e_layers=4, 100Hz: ~310ms — spans full PQRST complex."
        ),
    ),
    "TimesNet": ModelEntry(
        module="models.TimesNet",
        class_name="TimesNet",
        edge_compatible=False,
        layer_deps=["Embed", "Conv_Blocks"],
        notes="Strong detection (AUROC 0.970). 37.5MB, 15ms CPU — Orin Nano only.",
    ),
    "iTransformer": ModelEntry(
        module="models.iTransformer",
        class_name="iTransformer",
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        notes="Inverted attention: channels as tokens. Reserved for ECG+PPG multi-channel (enc_in=2).",
    ),
}


class _LazyModelDict:
    def get(self, name: str) -> type:
        if name not in _REGISTRY:
            raise KeyError(
                f"Model '{name}' not found. Available: {sorted(_REGISTRY.keys())}"
            )
        entry = _REGISTRY[name]
        try:
            module = importlib.import_module(entry.module)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import '{name}' from '{entry.module}'. "
                f"Layer deps: {entry.layer_deps}. Error: {exc}"
            ) from exc
        if not hasattr(module, entry.class_name):
            raise AttributeError(
                f"Module '{entry.module}' has no class '{entry.class_name}'"
            )
        return getattr(module, entry.class_name)

    def list_models(self) -> list[str]:
        return sorted(_REGISTRY.keys())

    def info(self, name: str) -> ModelEntry:
        if name not in _REGISTRY:
            raise KeyError(f"Model '{name}' not registered.")
        return _REGISTRY[name]


ModelRegistry = _LazyModelDict()
