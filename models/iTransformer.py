import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding_inverted


class iTransformer(nn.Module):
    """Inverted Transformer: variables as tokens, for anomaly detection."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        factor    = getattr(configs, "factor",     1)
        embed     = getattr(configs, "embed",      "fixed")
        freq      = getattr(configs, "freq",       "h")
        activation = getattr(configs, "activation", "gelu")

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, embed, freq, configs.dropout)
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
        self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)

    def forward(self, x):
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        _, L, N = x.shape

        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, L, 1)
        return dec_out
