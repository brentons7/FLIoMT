import torch
import torch.nn as nn
from models.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.layers.Embed import PatchEmbedding


class _Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        return x.transpose(*self.dims).contiguous() if self.contiguous else x.transpose(*self.dims)


class _FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten  = nn.Flatten(start_dim=-2)
        self.linear   = nn.Linear(nf, target_window)
        self.dropout  = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class PatchTST(nn.Module):
    """Patch-based Transformer for anomaly detection."""

    def __init__(self, configs, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.seq_len = configs.seq_len
        patch_len    = getattr(configs, "patch_len", patch_len)
        stride       = getattr(configs, "stride",    stride)
        factor       = getattr(configs, "factor",    1)
        activation   = getattr(configs, "activation", "gelu")
        padding      = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor, attention_dropout=configs.dropout,
                                  output_attention=False),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout, activation=activation,
             ) for _ in range(configs.e_layers)],
            norm_layer=nn.Sequential(
                _Transpose(1, 2), nn.BatchNorm1d(configs.d_model), _Transpose(1, 2)),
        )
        head_nf   = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = _FlattenHead(configs.enc_in, head_nf, configs.seq_len,
                                 head_dropout=configs.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalization
        means  = x.mean(1, keepdim=True).detach()
        x      = x - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x     /= stdev

        # Patch + encode per variable
        x      = x.permute(0, 2, 1)            # [B, C, L]
        enc_out, n_vars = self.patch_embedding(x)     # [B*C, patch_num, d_model]
        enc_out, _      = self.encoder(enc_out)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, d_model, patch_num]

        dec_out = self.head(enc_out).permute(0, 2, 1)  # [B, L, C]

        # De-normalization
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        return dec_out
