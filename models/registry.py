"""
Model registry with lazy loading and capability declarations.

Models are registered by name with their module path and metadata.
The actual model class is only imported when first requested, keeping
startup time low regardless of how many models are installed.

Source reference: tslib/exp/exp_basic.py (LazyModelDict)
Migration status: PARTIAL — registry structure defined; model entries to be
                  populated in Phase 3 as models are ported.

Usage:
    from models.registry import ModelRegistry

    model_cls = ModelRegistry.get("Transformer")
    model = model_cls(config)
"""

from __future__ import annotations
import importlib
from dataclasses import dataclass, field


@dataclass
class ModelEntry:
    """Metadata for a registered model."""
    module:           str                 # Import path (e.g., "models.transformer")
    class_name:       str                 # Class name within module (always "Model")
    tier:             int                 # Migration priority: 1 (highest) to 3 (lowest)
    edge_compatible:  bool                # Runs on Raspberry Pi without CUDA
    layer_deps:       list[str]           # Required files from models/layers/
    source:           str                 # tslib source file
    notes:            str = ""            # Extra context (notInUse status, caveats)


# Registry declaration — entries populated as models are ported in Phase 3.
# Models marked edge_compatible=False require CUDA or torch-geometric.
_REGISTRY: dict[str, ModelEntry] = {

    # -------------------------------------------------------------------------
    # Tier 1 — Migrate First
    # -------------------------------------------------------------------------
    "Transformer": ModelEntry(
        module="models.Transformer",
        class_name="Transformer",
        tier=1,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/Transformer.py",
        notes="Proven in FL on MSL dataset. Used in tslib/fl/ experiments.",
    ),
    "iTransformer": ModelEntry(
        module="models.iTransformer",
        class_name="iTransformer",
        tier=1,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/iTransformer.py",
        notes="Inverted attention: channels as tokens. Well-suited for multi-channel physio.",
    ),
    "PatchTST": ModelEntry(
        module="models.PatchTST",
        class_name="PatchTST",
        tier=1,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/PatchTST.py",
        notes="Patch windows align with cardiac cycle length at 100 Hz.",
    ),
    "KANAD": ModelEntry(
        module="models.KANAD",
        class_name="KANAD",
        tier=1,
        edge_compatible=True,
        layer_deps=[],
        source="tslib/models/KANAD.py",
        notes="Purpose-built for anomaly detection. Cosine basis suits periodic biosignals.",
    ),
    "Mamba2": ModelEntry(
        module="models.Mamba2",
        class_name="Mamba2",
        tier=1,
        edge_compatible=True,
        layer_deps=["Embed"],
        source="tslib/models/Mamba2.py",
        notes=(
            "Pure-PyTorch Mamba2 with parallel scan (O(log L)). "
            "No mamba_ssm / CUDA kernel — runs on Pi5/Nano/Xavier/CPU. "
            "Grouped heads with scalar A decay."
        ),
    ),
    "Autoformer": ModelEntry(
        module="models.Autoformer",
        class_name="Autoformer",
        tier=1,
        edge_compatible=True,
        layer_deps=["Embed", "AutoCorrelation", "Autoformer_EncDec"],
        source="tslib/models/Autoformer.py",
        notes="Series decomposition (trend + seasonal) directly relevant to HRV.",
    ),
    "Nonstationary_Transformer": ModelEntry(
        module="models.Nonstationary_Transformer",
        class_name="Nonstationary_Transformer",
        tier=1,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/Nonstationary_Transformer.py",
        notes=(
            "De-stationary projection handles amplitude/frequency shifts "
            "across activity levels — critical for physiological signals."
        ),
    ),
    "DLinear": ModelEntry(
        module="models.DLinear",
        class_name="DLinear",
        tier=1,
        edge_compatible=True,
        layer_deps=["Autoformer_EncDec"],
        source="tslib/models/DLinear.py",
        notes="Simple linear baseline. Essential lower-bound comparison.",
    ),

    # -------------------------------------------------------------------------
    # Tier 2 — Extended Comparison Set
    # -------------------------------------------------------------------------
    "TimesNet": ModelEntry(
        module="models.TimesNet",
        class_name="TimesNet",
        tier=2,
        edge_compatible=True,
        layer_deps=["Embed", "Conv_Blocks"],
        source="tslib/models/TimesNet.py",
        notes="2D temporal variation via FFT period discovery. High-priority.",
    ),
    "TimeMixer": ModelEntry(
        module="models.TimeMixer",
        class_name="TimeMixer",
        tier=2,
        edge_compatible=True,
        layer_deps=["Embed", "Autoformer_EncDec", "StandardNorm"],
        source="tslib/models/TimeMixer.py",
        notes=(
            "Multi-scale decomposable mixing. Strong on non-stationary physio signals. "
            "Use down_sampling_layers=0 for single-scale AD (default)."
        ),
    ),
    "Reformer": ModelEntry(
        module="models.Reformer",
        class_name="Reformer",
        tier=2,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/Reformer.py",
        notes="LSH attention; memory-efficient for long sequences. Not yet ported.",
    ),
    "SegRNN": ModelEntry(
        module="models.SegRNN",
        class_name="SegRNN",
        tier=2,
        edge_compatible=True,
        layer_deps=[],
        source="tslib/models/SegRNN.py",
        notes="Segment-based RNN. Suitable for causal/streaming inference. Not yet ported.",
    ),
    "FiLM": ModelEntry(
        module="models.FiLM",
        class_name="FiLM",
        tier=2,
        edge_compatible=True,
        layer_deps=[],
        source="tslib/models/FiLM.py",
        notes="Legendre Memory Units for long-range dependency. Not yet ported.",
    ),
    "MICN": ModelEntry(
        module="models.MICN",
        class_name="MICN",
        tier=2,
        edge_compatible=True,
        layer_deps=["Embed", "Autoformer_EncDec"],
        source="tslib/models/MICN.py",
        notes="Multi-scale isometric convolution. Waveform-friendly. Not yet ported.",
    ),
    "Informer": ModelEntry(
        module="models.Informer",
        class_name="Informer",
        tier=2,
        edge_compatible=True,
        layer_deps=["Transformer_EncDec", "SelfAttention_Family", "Embed"],
        source="tslib/models/Informer.py",
        notes="ProbSparse attention. Standard literature baseline. Not yet ported.",
    ),

    # -------------------------------------------------------------------------
    # Tier 3 — Experimental / Complex Dependencies
    # -------------------------------------------------------------------------
    "MambaSimple": ModelEntry(
        module="models.mamba_simple",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["embed"],
        source="tslib/models/MambaSimple.py",
        notes="SSM with einops. Useful for SSM architecture comparison.",
    ),
    "LightTS": ModelEntry(
        module="models.lightts",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=[],
        source="tslib/models/LightTS.py",
        notes="Lightweight MLP-Mixer. Useful for edge-only inference.",
    ),
    "ETSformer": ModelEntry(
        module="models.etsformer",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["embed", "etsformer_enc_dec"],
        source="tslib/models/ETSformer.py",
        notes="Exponential smoothing. Better suited to stationary signals.",
    ),
    "Crossformer": ModelEntry(
        module="models.crossformer",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["crossformer_enc_dec", "embed", "attention"],
        source="tslib/models/Crossformer.py",
        notes="Cross-dimension attention. Useful if ECG/PPG channel interactions matter.",
    ),
    "FEDformer": ModelEntry(
        module="models.fedformer",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["embed", "autocorrelation", "fourier_correlation",
                    "multiwavelet_correlation", "autoformer_enc_dec"],
        source="tslib/models/FEDformer.py",
        notes="Most complex layer dependency set. Frequency-domain research value.",
    ),
    "Pyraformer": ModelEntry(
        module="models.pyraformer",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["pyraformer_enc_dec"],
        source="tslib/models/Pyraformer.py",
        notes="Pyramidal attention. Memory-efficient for very long sequences.",
    ),
    "TimeFilter": ModelEntry(
        module="models.timefilter",
        class_name="Model",
        tier=3,
        edge_compatible=True,
        layer_deps=["embed", "standard_norm", "timefilter_layers"],
        source="tslib/models/TimeFilter.py",
        notes="MoE routing. Experimental; alpha/top_p require tuning.",
    ),
    "MSGNet": ModelEntry(
        module="models.msgnet",
        class_name="Model",
        tier=3,
        edge_compatible=False,
        layer_deps=["embed", "msg_block"],
        source="tslib/models/MSGNet.py",
        notes=(
            "Graph convolution (GCN). Only useful when sensor topology forms a "
            "meaningful graph. Defer until multi-sensor wearable network is deployed."
        ),
    ),
}

# Models explicitly not supported for anomaly detection in tslib:
_UNSUPPORTED: dict[str, str] = {
    "FreTS":          "forward() raises ValueError for non-forecast tasks despite __init__ branch",
    "TiDE":           "raises NotImplementedError for anomaly_detection",
    "MultiPatchFormer": "raises NotImplementedError for anomaly_detection",
    "WPMixer":        "raises NotImplementedError for anomaly_detection",
}


class _LazyModelDict:
    """
    Lazy-loading model dictionary.

    Imports the model module only when the model class is first requested.
    All models expose a 'Model' class as their public interface.
    """

    def get(self, name: str) -> type:
        """
        Retrieve a model class by name.

        Args:
            name: Registered model name (e.g., "Transformer")

        Returns:
            The model class (uninstantiated)

        Raises:
            KeyError: If the model name is not registered
            NotImplementedError: If the model module has not been ported yet
            ImportError: If required dependencies are missing
        """
        if name in _UNSUPPORTED:
            raise ValueError(
                f"Model '{name}' does not support anomaly detection: {_UNSUPPORTED[name]}"
            )
        if name not in _REGISTRY:
            available = sorted(_REGISTRY.keys())
            raise KeyError(
                f"Model '{name}' not found. Available models: {available}"
            )
        entry = _REGISTRY[name]
        try:
            module = importlib.import_module(entry.module)
        except ImportError as exc:
            raise ImportError(
                f"Failed to import model '{name}' from '{entry.module}'. "
                f"Is the model ported yet? Dependencies: {entry.layer_deps}. "
                f"Original error: {exc}"
            ) from exc
        if not hasattr(module, entry.class_name):
            raise AttributeError(
                f"Module '{entry.module}' has no class '{entry.class_name}'"
            )
        return getattr(module, entry.class_name)

    def list_models(self, tier: int | None = None, edge_only: bool = False) -> list[str]:
        """
        List registered model names with optional filtering.

        Args:
            tier:      If given, return only models of this tier (1, 2, or 3)
            edge_only: If True, return only edge-compatible models

        Returns:
            Sorted list of model names
        """
        entries = _REGISTRY.items()
        if tier is not None:
            entries = ((k, v) for k, v in entries if v.tier == tier)
        if edge_only:
            entries = ((k, v) for k, v in entries if v.edge_compatible)
        return sorted(k for k, _ in entries)

    def info(self, name: str) -> ModelEntry:
        """Return the ModelEntry metadata for a registered model."""
        if name not in _REGISTRY:
            raise KeyError(f"Model '{name}' not registered.")
        return _REGISTRY[name]


ModelRegistry = _LazyModelDict()
