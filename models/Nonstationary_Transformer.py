import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.SelfAttention_Family import DSAttention, AttentionLayer
from models.layers.Embed import DataEmbedding


class _Projector(nn.Module):
    """MLP that learns de-stationary factors (tau and delta)."""

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(seq_len, 1, kernel_size, padding=padding,
                                     padding_mode="circular", bias=False)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        batch_size = x.shape[0]
        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)
        return self.backbone(x)


class Nonstationary_Transformer(nn.Module):
    """Non-stationary Transformer with learned de-stationarization for anomaly detection."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len     = configs.seq_len
        factor           = getattr(configs, "factor",          1)
        embed            = getattr(configs, "embed",           "fixed")
        freq             = getattr(configs, "freq",            "h")
        activation       = getattr(configs, "activation",      "gelu")
        p_hidden_dims    = getattr(configs, "p_hidden_dims",   [128, 128])
        p_hidden_layers  = getattr(configs, "p_hidden_layers", 2)

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    DSAttention(False, factor, attention_dropout=configs.dropout,
                                output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=activation,
             ) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.projection    = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.tau_learner   = _Projector(configs.enc_in, configs.seq_len,
                                        p_hidden_dims, p_hidden_layers, output_dim=1)
        self.delta_learner = _Projector(configs.enc_in, configs.seq_len,
                                        p_hidden_dims, p_hidden_layers, output_dim=configs.seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw   = x.clone().detach()
        mean_enc = x.mean(1, keepdim=True).detach()
        x        = x - mean_enc
        std_enc  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x        = x / std_enc

        tau   = self.tau_learner(x_raw, std_enc)
        tau   = torch.exp(torch.clamp(tau, max=80.0))
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)
        dec_out = self.projection(enc_out)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out
