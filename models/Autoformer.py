import torch
import torch.nn as nn
from models.layers.Embed import DataEmbedding_wo_pos
from models.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """Autoformer with auto-correlation for anomaly detection."""

    def __init__(self, configs):
        super().__init__()
        factor     = getattr(configs, "factor",     1)
        embed      = getattr(configs, "embed",      "fixed")
        freq       = getattr(configs, "freq",       "h")
        activation = getattr(configs, "activation", "gelu")
        moving_avg = getattr(configs, "moving_avg", 25)

        self.decomp = series_decomp(moving_avg)
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, factor, attention_dropout=configs.dropout,
                                    output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                moving_avg=moving_avg, dropout=configs.dropout, activation=activation,
             ) for _ in range(configs.e_layers)],
            norm_layer=my_Layernorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(enc_out)
