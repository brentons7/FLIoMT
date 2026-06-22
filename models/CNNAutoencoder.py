import torch
import torch.nn as nn


class _ResidualDilatedBlock(nn.Module):
    """
    Residual dilated Conv1d block: Conv → BatchNorm → GELU → Dropout + skip.

    Padding = dilation (for kernel_size=3) preserves sequence length exactly,
    so no positional information is lost regardless of dilation depth.
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.conv    = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.norm    = nn.BatchNorm1d(channels)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.norm(self.conv(x))))


class CNNAutoencoder(nn.Module):
    """
    1D dilated CNN autoencoder for reconstruction-based anomaly detection.

    Architecture:
        input projection  enc_in → d_model  (Conv1d kernel=1)
        e_layers residual dilated Conv1d blocks, dilation = 2^i per layer
        output projection d_model → c_out   (Conv1d kernel=1)

    Temporal resolution is preserved throughout — no strided downsampling.
    Sharp transients (QRS spikes) are never averaged away by the architecture.

    Receptive field grows as: RF = 1 + sum(2 * 2^i for i in 0..e_layers-1)
    At e_layers=4, 100 Hz:  RF = 31 samples = 310 ms  (covers PQRST complex)
    At e_layers=5, 100 Hz:  RF = 63 samples = 630 ms  (covers full cardiac cycle)

    Uses d_model for channel width; d_ff and n_heads are ignored (kept in
    configs for CLI/YAML compatibility with the rest of the model registry).

    Interface: forward(x: [B, L, C]) → x_hat: [B, L, C]
    """

    def __init__(self, configs):
        super().__init__()
        enc_in   = configs.enc_in
        c_out    = configs.c_out
        d_model  = configs.d_model
        e_layers = configs.e_layers
        dropout  = configs.dropout

        self.input_proj  = nn.Conv1d(enc_in, d_model, kernel_size=1, bias=False)
        self.blocks      = nn.ModuleList([
            _ResidualDilatedBlock(d_model, dilation=2 ** i, dropout=dropout)
            for i in range(e_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects [B, C, L]; permute in and out
        z = self.input_proj(x.permute(0, 2, 1))  # [B, d_model, L]
        for block in self.blocks:
            z = block(z)
        return self.output_proj(z).permute(0, 2, 1)  # [B, L, C]
