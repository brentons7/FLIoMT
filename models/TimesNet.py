import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from models.layers.Embed import DataEmbedding
from models.layers.Conv_Blocks import Inception_Block_V1


def _fft_for_period(x: torch.Tensor, k: int = 2):
    """
    Discover top-k dominant periods via FFT amplitude spectrum.

    Args:
        x: [B, T, C]
        k: number of top periods to return

    Returns:
        period_list: numpy array of k period lengths
        period_weight: [B, k] amplitude weights for adaptive aggregation
    """
    xf             = torch.fft.rfft(x, dim=1)
    freq_list      = abs(xf).mean(0).mean(-1)
    freq_list[0]   = 0                          # zero DC component
    _, top_list    = torch.topk(freq_list, k)
    top_list       = top_list.detach().cpu().numpy()
    period         = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class _TimesBlock(nn.Module):
    """
    Convert 1D temporal signal to 2D via period reshaping, then apply 2D Inception convs.
    Each discovered period becomes the height of a 2D map; time steps become the width.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len  = configs.seq_len
        self.pred_len = getattr(configs, "pred_len", 0)
        self.k        = getattr(configs, "top_k", 5)
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=getattr(configs, "num_kernels", 6)),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=getattr(configs, "num_kernels", 6)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N       = x.size()
        total_len     = self.seq_len + self.pred_len   # == seq_len for AD (pred_len=0)
        period_list, period_weight = _fft_for_period(x, self.k)

        res = []
        for period in period_list:
            if period == 0:
                period = 1
            if total_len % period != 0:
                pad_len = ((total_len // period) + 1) * period - total_len
                padding = torch.zeros(B, pad_len, N, device=x.device, dtype=x.dtype)
                out     = torch.cat([x, padding], dim=1)
                length  = total_len + pad_len
            else:
                out    = x
                length = total_len

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :total_len, :])

        res            = torch.stack(res, dim=-1)                      # [B, T, N, k]
        period_weight  = F.softmax(period_weight, dim=1)              # [B, k]
        period_weight  = period_weight.unsqueeze(1).unsqueeze(1)       # [B, 1, 1, k]
        period_weight  = period_weight.expand(B, T, N, -1)
        res            = (res * period_weight).sum(-1)                 # [B, T, N]
        return res + x                                                 # residual


class TimesNet(nn.Module):
    """
    TimesNet: temporal 2D variation modeling for anomaly detection.

    Key idea: FFT-discovered dominant periods reshape the 1D time series into
    a 2D map, where intra-period (column) and inter-period (row) variations
    are captured by 2D Inception convolutions. Adaptive aggregation over k
    periods gives a strong multi-scale representation.

    Source: tslib/models/TimesNet.py — stripped to anomaly_detection task.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len     = configs.seq_len
        self.pred_len    = getattr(configs, "pred_len", 0)
        embed            = getattr(configs, "embed",    "fixed")
        freq             = getattr(configs, "freq",     "h")

        self.blocks = nn.ModuleList([
            _TimesBlock(configs) for _ in range(configs.e_layers)
        ])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.layer_norm    = nn.LayerNorm(configs.d_model)
        self.projection    = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Non-stationary normalization
        means  = x.mean(1, keepdim=True).detach()
        x      = x - means
        stdev  = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x      = x / stdev

        enc_out = self.enc_embedding(x, None)
        for block in self.blocks:
            enc_out = self.layer_norm(block(enc_out))

        dec_out = self.projection(enc_out)

        # De-normalize
        total = self.seq_len + self.pred_len
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, total, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, total, -1)
        return dec_out
