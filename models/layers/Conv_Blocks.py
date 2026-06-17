"""Inception-style 2D conv blocks used by TimesNet."""

import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """
    Parallel multi-scale 2D convolutions averaging over kernel ensemble.
    Used by TimesBlock to capture temporal patterns at different granularities.

    Source: tslib/layers/Conv_Blocks.py (copied verbatim)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_kernels: int = 6, init_weight: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_kernels  = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([k(x) for k in self.kernels], dim=-1).mean(-1)
