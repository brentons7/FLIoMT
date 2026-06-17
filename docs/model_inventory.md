# Model Inventory

All models are ported from `../tslib/models/`. They are grouped into three
migration tiers. Within each tier, models are listed in approximate migration
priority order.

## Interface Contract

Every model in FLIoMT must satisfy:

```python
# Input:  x    — float32 Tensor [batch, seq_len, n_channels]
# Output: x_hat — float32 Tensor [batch, seq_len, n_channels]  (reconstruction)
model = MyModel(config)
x_hat = model(x)
assert x_hat.shape == x.shape
```

The TSlib 4-argument forward signature `model(x, x_mark, dec_inp, dec_mark)`
is simplified to `model(x)` in FLIoMT. Unused arguments are dropped at the
call site, not by adding None-accepting overloads.

---

## Tier 1 — Migrate First (8 models)

Core experimental set. These models are either well-established for anomaly
detection in the TSlib benchmarks or have specific relevance to edge deployment
and physiological signal processing.

| Model | tslib source | Edge | Layer deps | Notes |
|-------|-------------|------|-----------|-------|
| Transformer | `models/Transformer.py` | Yes | Embed.py, Transformer_EncDec.py, SelfAttention_Family.py | Baseline; MSL seq_len=100, d_model=64 |
| iTransformer | `models/iTransformer.py` | Yes | Embed.py, Transformer_EncDec.py, SelfAttention_Family.py | Inverted attention — variables as tokens; d_model=32 |
| PatchTST | `models/PatchTST.py` | Yes | PatchTST_backbone.py | Patch-based; d_model=32, patch_len=16 |
| KANAD | `models/KANAD.py` | Yes | None (self-contained) | KAN basis functions; d_model=3 (basis order, not hidden dim) |
| Mamba2 | `models/Mamba2.py` | Yes | Embed.py | Pure-PyTorch Mamba2 with parallel scan; no mamba_ssm / CUDA kernel; Pi+Nano+Xavier compatible |
| Autoformer | `models/Autoformer.py` | Yes | Autoformer_EncDec.py, Embed.py, Correlation.py | Auto-correlation; moving_avg=25 |
| NonStationary_Transformer | `models/Nonstationary_Transformer.py` | Yes | Embed.py, Transformer_EncDec.py, NSTransformer.py | Stationarization layer; for non-stationary physiological signals |
| DLinear | `models/DLinear.py` | Yes | None | Decomposition linear; lightweight |

**Mamba2 note**: Pure-PyTorch implementation using a parallel tree-reduction scan
instead of the sequential loop used in Mamba 1. No `mamba_ssm` package required —
runs on all Pi/Nano/Xavier hardware without CUDA. Uses grouped heads with scalar A
decay per (head, state). `d_ff` in the config doubles as `d_state`.

---

## Tier 2 — Extended Comparison (8 models)

Strong models for cross-architecture benchmarking. Migrate after Tier 1
infrastructure is validated.

| Model | tslib source | Edge | Layer deps | Notes |
|-------|-------------|------|-----------|-------|
| TimeMixer | `models/TimeMixer.py` | Yes | Embed.py | Multiscale mixing; down_sampling_layers=2, down_sampling_window=2 |
| SegRNN | `models/SegRNN.py` | Yes | None | Segment-level RNN; no attention overhead |
| Crossformer | `models/Crossformer.py` | Yes | Crossformer_EncDec.py, PatchTST_backbone.py | Cross-dimension attention |
| SCINet | `models/SCINet.py` | Yes | SCINet_backbone.py | Iterative convolutional decomposition |
| TimesNet | `models/TimesNet.py` | Yes | Inception_Block.py | 2D temporal transforms; Conv-based |
| FEDformer | `models/FEDformer.py` | Yes | FEDformer_backbone.py, ETSformer_EncDec.py | Frequency-domain attention |
| Informer | `models/Informer.py` | Yes | Embed.py, SelfAttention_Family.py, Transformer_EncDec.py | ProbSparse attention |
| ETSformer | `models/ETSformer.py` | Yes | ETSformer_EncDec.py | Exponential smoothing + Transformer |

---

## Tier 3 — Experimental (8 models)

Complex dependencies, niche use cases, or heavier compute. Migrate after
Tier 2 results are analyzed.

| Model | tslib source | Edge | Layer deps | Notes |
|-------|-------------|------|-----------|-------|
| PatchMixer | `models/PatchMixer.py` | Yes | None | Patch-based MLP mixer |
| MSGNet | `models/MSGNet.py` | No | MSGBlock.py (graph layers) | Multiscale graph network; heavy |
| TimeFilter | `models/TimeFilter.py` | Yes | TimeFilter_layers.py (MoE) | Mixture-of-experts filter |
| SOFTS | `models/SOFTS.py` | Yes | Embed.py | Star aggregate for multi-variate |
| Reformer | `models/Reformer.py` | Yes | reformer_enc.py | LSH attention; unusual dep |
| MICN | `models/MICN.py` | Yes | MICN_backbone.py | Multi-scale isometric convolution |
| MambaSimple | `models/MambaSimple.py` | No | mamba_ssm (CUDA kernel) | Requires mamba_ssm package; not Pi-compatible |

**MambaSimple note**: Uses the official `mamba_ssm` package with CUDA kernels.
Does not run on Raspberry Pi or ARM without CUDA. Tier 3 reflects the dependency
constraint. Use Mamba2 (Tier 1) for device-portable SSM experiments.

---

## Unsupported Models (do not migrate)

These models exist in `tslib/models/` with `anomaly_detection` method stubs
but raise exceptions at runtime. They are documented here to avoid re-testing.

| Model | Reason | tslib source |
|-------|--------|-------------|
| FreTS | **Runtime bug**: `forward()` raises `ValueError('Only forecast tasks implemented yet')` despite having an `anomaly_detection()` method. The `__init__` branch exists but the implementation was never completed. | `models/FreTS.py` |
| TiDE | `raise NotImplementedError("Task anomaly_detection for Tide is temporarily not supported")` | `models/TiDE.py` |
| MultiPatchFormer | `raise NotImplementedError("Task anomaly_detection for WPMixer is temporarily not supported")` — note the error message says WPMixer; copy-paste bug | `models/MultiPatchFormer.py` |
| WPMixer | `raise NotImplementedError("Task anomaly_detection for WPMixer is temporarily not supported")` | `models/WPMixer.py` |

---

## Shared Layer Dependencies

Most models pull from `tslib/layers/`. The migration copies only the layers
each model needs. Layers used by multiple models are shared.

| Layer file | Used by |
|-----------|---------|
| `Embed.py` | Transformer, iTransformer, Autoformer, NonStationary_Transformer, FEDformer, Informer, TimeMixer, SOFTS |
| `Transformer_EncDec.py` | Transformer, iTransformer, NonStationary_Transformer, Informer |
| `SelfAttention_Family.py` | Transformer, iTransformer, NonStationary_Transformer, Informer |
| `Autoformer_EncDec.py` | Autoformer |
| `Correlation.py` | Autoformer |
| `PatchTST_backbone.py` | PatchTST, Crossformer |
| `Crossformer_EncDec.py` | Crossformer |
| `ETSformer_EncDec.py` | ETSformer, FEDformer |
| `FEDformer_backbone.py` | FEDformer |
| `SCINet_backbone.py` | SCINet |
| `Inception_Block.py` | TimesNet |
| `MSGBlock.py` | MSGNet |
| `TimeFilter_layers.py` | TimeFilter |
| `NSTransformer.py` | NonStationary_Transformer |
| `MICN_backbone.py` | MICN |

---

## Model Registry

`models/registry.py` exports a `ModelRegistry` singleton with lazy loading.
Use it as follows:

```python
from models.registry import ModelRegistry

# List all available models
ModelRegistry.list_models()

# List only edge-compatible Tier 1 models
ModelRegistry.list_models(tier=1, edge_only=True)

# Get model info without loading weights
ModelRegistry.info("Transformer")

# Instantiate model (imports module on first access)
model_cls = ModelRegistry.get("Transformer")
model = model_cls(config)

# Check if a model is supported
if "FreTS" in ModelRegistry._UNSUPPORTED:
    print(ModelRegistry._UNSUPPORTED["FreTS"])
```

---

## notInUse/ Context

The tslib repository contains `scripts/anomaly_detection/MSL/notInUse/` and
`scripts/anomaly_detection/PSM/notInUse/` directories with scripts for models
that were benchmarked on centralized datasets and shelved. The models present
there but not in FLIoMT's migration scope are:

Crossformer, DLinear, ETSformer, FEDformer, FiLM, Informer, LightTS,
MambaSimple, MICN, Pyraformer, Reformer

These appear in `notInUse/` because they performed less well than Tier 1
models on centralized MSL/PSM benchmarks. They remain in FLIoMT's Tier 2/3
lists because:
1. Cross-architecture experimentation is a stated research goal
2. Centralized benchmark performance does not predict FL or physio performance
3. Hardware constraints differ (some Pi-capable models were shelved for
   accuracy reasons, not compute reasons)

No `notInUse/` models are permanently excluded — only the four models that
raise `NotImplementedError` or have runtime bugs.
