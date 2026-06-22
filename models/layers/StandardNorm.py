"""Reversible instance normalization."""

import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Reversible instance normalization (RevIN-style) with optional affine params.

    Usage:
        norm = Normalize(num_features=C, affine=True)
        x_normed = norm(x, 'norm')   # stores stats, normalizes
        x_orig   = norm(x_normed, 'denorm')  # restores original scale

    Source: tslib/layers/StandardNorm.py (copied verbatim)
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 affine: bool = False, subtract_last: bool = False,
                 non_norm: bool = False):
        super().__init__()
        self.num_features  = num_features
        self.eps           = eps
        self.affine        = affine
        self.subtract_last = subtract_last
        self.non_norm      = non_norm
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown mode: {mode!r}. Use 'norm' or 'denorm'.")

    def _get_statistics(self, x: torch.Tensor) -> None:
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x
