import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from models.layers.Embed import DataEmbedding


class Informer(nn.Module):
    """
    Informer with ProbSparse attention (O(L log L)) for anomaly detection.
    Encoder-only: enc_embedding → ProbAttention encoder → linear projection.
    No decoder or distilling ConvLayer (those are forecast-only in tslib).

    Source: tslib/models/Informer.py — stripped to anomaly_detection task.
    """

    def __init__(self, configs):
        super().__init__()
        factor     = getattr(configs, "factor",     5)
        embed      = getattr(configs, "embed",      "fixed")
        freq       = getattr(configs, "freq",       "h")
        activation = getattr(configs, "activation", "gelu")

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    ProbAttention(False, factor, attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=activation,
             ) for _ in range(configs.e_layers)],
            conv_layers=None,
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)
