import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding


class Transformer(nn.Module):
    """Vanilla Transformer autoencoder for anomaly detection."""

    def __init__(self, configs):
        super().__init__()
        factor    = getattr(configs, "factor",     1)
        embed     = getattr(configs, "embed",      "fixed")
        freq      = getattr(configs, "freq",       "h")
        activation = getattr(configs, "activation", "gelu")

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=activation,
             ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x):
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)
