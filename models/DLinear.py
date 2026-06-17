import torch
import torch.nn as nn
from models.layers.Autoformer_EncDec import series_decomp


class DLinear(nn.Module):
    """Decomposition Linear for anomaly detection."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        moving_avg = getattr(configs, "moving_avg", 25)
        individual = getattr(configs, "individual", False)

        self.decomposition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.seq_len) for _ in range(self.channels)])
            self.Linear_Trend    = nn.ModuleList(
                [nn.Linear(self.seq_len, self.seq_len) for _ in range(self.channels)])
            for i in range(self.channels):
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
                self.Linear_Trend[i].weight    = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Trend    = nn.Linear(self.seq_len, self.seq_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))
            self.Linear_Trend.weight    = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.seq_len, self.seq_len]))

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)  # [B, C, L]
        trend_init    = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_out = torch.zeros_like(seasonal_init)
            trend_out    = torch.zeros_like(trend_init)
            for i in range(self.channels):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_out[:, i, :]    = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal_init)
            trend_out    = self.Linear_Trend(trend_init)

        return (seasonal_out + trend_out).permute(0, 2, 1)  # [B, L, C]
