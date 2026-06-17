"""
Shared layer implementations used by anomaly detection models.

Source: tslib/layers/
Migration status: PENDING — layers to be ported in Phase 3 alongside models.

Planned modules:
    embed                    — DataEmbedding, PatchEmbedding, DataEmbedding_inverted
    transformer_enc_dec      — Encoder, Decoder, EncoderLayer, DecoderLayer, ConvLayer
    attention                — FullAttention, ProbAttention, ReformerLayer, AttentionLayer
    autoformer_enc_dec       — series_decomp, AutoCorrelationLayer blocks
    autocorrelation          — AutoCorrelation mechanism
    conv_blocks              — Inception_Block_V1 (for TimesNet)
    crossformer_enc_dec      — TwoStageAttentionLayer, scale_block
    etsformer_enc_dec        — ExponentialSmoothing components
    fourier_correlation      — FourierBlock, FourierCrossAttention
    multiwavelet_correlation — MultiWaveletTransform, MultiWaveletCross
    pyraformer_enc_dec       — Pyramidal attention encoder
    mamba_block              — Pure-PyTorch SSM block
    msg_block                — Graph convolution blocks (MSGNet)
    standard_norm            — Normalize wrapper
    timefilter_layers        — TimeFilter_Backbone, MoE routing
"""
