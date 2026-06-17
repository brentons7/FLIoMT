import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class _KANADBlock(nn.Module):
    """Kolmogorov-Arnold basis function block (self-contained, no layer deps)."""

    def __init__(self, window: int, order: int):
        super().__init__()
        self.order = order
        self.window = window
        self.channels = 2 * order + 1
        self.register_buffer(
            "orders",
            self._cosine_basis(window, order).unsqueeze(0),  # [1, order, window]
        )
        self.out_conv   = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act        = nn.GELU()
        self.bn1        = nn.BatchNorm1d(self.channels)
        self.bn2        = nn.BatchNorm1d(self.channels)
        self.bn3        = nn.BatchNorm1d(1)
        self.init_conv  = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(window, window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [x.unsqueeze(1)]
        ff = torch.cat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [torch.cos(o * x.unsqueeze(1)) for o in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )
        res.append(ff)
        ff = self.act(self.bn1(self.init_conv(ff)))
        ff = self.act(self.bn2(self.inner_conv(ff) + res.pop()))
        ff = self.act(self.bn3(self.out_conv(ff) + res.pop()))
        return self.final_conv(ff).squeeze(1)

    @staticmethod
    def _cosine_basis(window: int, period) -> torch.Tensor:
        d  = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else list(range(1, period + 1))
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i] = torch.cos(t)
        return result


class KANAD(nn.Module):
    """
    Kolmogorov-Arnold Network for Anomaly Detection.

    configs.d_model is the KAN basis order (e.g., 3), NOT a transformer hidden dim.
    Self-contained: no shared layer dependencies.
    """

    def __init__(self, configs):
        super().__init__()
        self.enc = _KANADBlock(window=configs.seq_len, order=configs.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] → process each channel independently
        x_in = rearrange(x, "B L D -> (B D) L")
        out  = self.enc(x_in)
        return rearrange(out, "(B D) L -> B L D", B=x.size(0))
