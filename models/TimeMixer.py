import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Embed import DataEmbedding_wo_pos
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.StandardNorm import Normalize


class _DFT_series_decomp(nn.Module):
    """FFT-based trend/season decomposition."""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        xf            = torch.fft.rfft(x, dim=1)
        freq          = abs(xf)
        freq[0]       = 0
        top_k_freq, _ = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season  = torch.fft.irfft(xf, dim=1)
        return x_season, x - x_season


class _MultiScaleSeasonMixing(nn.Module):
    """Bottom-up cross-scale season blending."""

    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** (i + 1))),
            )
            for i in range(configs.down_sampling_layers)
        ])

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out_high       = season_list[0]
        out_season_out = [out_high.permute(0, 2, 1)]
        for i, layer in enumerate(self.layers):
            out_low_res = layer(out_high)
            out_high    = season_list[i + 1] + out_low_res
            out_season_out.append(out_high.permute(0, 2, 1))
        return out_season_out


class _MultiScaleTrendMixing(nn.Module):
    """Top-down cross-scale trend blending."""

    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                          configs.seq_len // (configs.down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(configs.seq_len // (configs.down_sampling_window ** i),
                          configs.seq_len // (configs.down_sampling_window ** i)),
            )
            for i in reversed(range(configs.down_sampling_layers))
        ])

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        rev           = list(reversed(trend_list))
        out_low       = rev[0]
        out_trend_out = [out_low.permute(0, 2, 1)]
        for i, layer in enumerate(self.layers):
            out_high_res = layer(out_low)
            out_low      = rev[i + 1] + out_high_res
            out_trend_out.append(out_low.permute(0, 2, 1))
        out_trend_out.reverse()
        return out_trend_out


class _PastDecomposableMixing(nn.Module):
    """Core PDM block: decompose → mix season + trend → recombine."""

    def __init__(self, configs):
        super().__init__()
        self.channel_independence = configs.channel_independence
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout    = nn.Dropout(configs.dropout)

        decomp_method = getattr(configs, "decomp_method", "moving_avg")
        moving_avg    = getattr(configs, "moving_avg",    25)
        top_k         = getattr(configs, "top_k",          5)
        if decomp_method == "dft_decomp":
            self.decomp = _DFT_series_decomp(top_k)
        else:
            self.decomp = series_decomp(moving_avg)

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_ff),
                nn.GELU(),
                nn.Linear(configs.d_ff, configs.d_model),
            )

        self.season_mixing = _MultiScaleSeasonMixing(configs)
        self.trend_mixing  = _MultiScaleTrendMixing(configs)
        self.out_cross     = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Linear(configs.d_ff, configs.d_model),
        )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        lengths      = [x.size(1) for x in x_list]
        season_list, trend_list = [], []
        for x in x_list:
            s, t = self.decomp(x)
            if not self.channel_independence:
                s = self.cross_layer(s)
                t = self.cross_layer(t)
            season_list.append(s.permute(0, 2, 1))
            trend_list.append(t.permute(0, 2, 1))

        out_seasons = self.season_mixing(season_list)
        out_trends  = self.trend_mixing(trend_list)

        out_list = []
        for ori, s, t, L in zip(x_list, out_seasons, out_trends, lengths):
            combined = s + t
            if self.channel_independence:
                out = ori + self.out_cross(combined)
            else:
                out = combined
            out_list.append(out[:, :L, :])
        return out_list


class TimeMixer(nn.Module):
    """
    TimeMixer: multi-scale decomposable mixing for anomaly detection.

    Operates on multiple temporal scales (if down_sampling_layers > 0) via
    past decomposable mixing (PDM) blocks that blend season and trend components
    bottom-up and top-down across scales.

    For single-scale (default AD config): down_sampling_layers=0, which degrades
    gracefully to a single-scale decomp+mix architecture.

    Source: tslib/models/TimeMixer.py — stripped to anomaly_detection task;
    fixed __multi_scale_process_inputs to always return a list.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len               = configs.seq_len
        self.enc_in                = configs.enc_in
        self.down_sampling_layers  = getattr(configs, "down_sampling_layers",  0)
        self.down_sampling_window  = getattr(configs, "down_sampling_window",  1)
        self.down_sampling_method  = getattr(configs, "down_sampling_method",  None)
        self.channel_independence  = getattr(configs, "channel_independence",  1)
        use_norm                   = getattr(configs, "use_norm",               1)
        embed                      = getattr(configs, "embed",                "fixed")
        freq                       = getattr(configs, "freq",                  "h")

        self.pdm_blocks = nn.ModuleList([
            _PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])
        self.preprocess = series_decomp(getattr(configs, "moving_avg", 25))

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                1, configs.d_model, embed, freq, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                configs.enc_in, configs.d_model, embed, freq, configs.dropout)

        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=(use_norm == 0))
            for _ in range(self.down_sampling_layers + 1)
        ])

        if self.channel_independence:
            self.projection = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def _multi_scale_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Produce a list of tensors at decreasing temporal resolutions.
        Always returns a list (single-element if no downsampling).
        """
        if self.down_sampling_method is None or self.down_sampling_layers == 0:
            return [x]

        if self.down_sampling_method == "max":
            pool = nn.MaxPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "avg":
            pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            pool = nn.Conv1d(
                self.enc_in, self.enc_in,
                kernel_size=3, padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular", bias=False,
            ).to(x.device)
        else:
            return [x]

        scales   = [x]
        x_perm   = x.permute(0, 2, 1)  # B,C,T for pooling
        for _ in range(self.down_sampling_layers):
            x_perm = pool(x_perm)
            scales.append(x_perm.permute(0, 2, 1))
        return scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N  = x.size()
        scales   = self._multi_scale_inputs(x)

        # Normalize and optionally reshape for channel independence
        x_list = []
        for i, s in enumerate(scales):
            s = self.normalize_layers[i](s, "norm")
            if self.channel_independence:
                s = s.permute(0, 2, 1).contiguous().reshape(B * N, s.size(1), 1)
            x_list.append(s)

        # Embed each scale
        enc_list = [self.enc_embedding(s, None) for s in x_list]

        # PDM blocks over all scales
        for block in self.pdm_blocks:
            enc_list = block(enc_list)

        # Project from finest scale
        dec = self.projection(enc_list[0])  # [B*N, T, 1] or [B, T, c_out]

        if self.channel_independence:
            dec = dec.reshape(B, N, -1).permute(0, 2, 1).contiguous()  # [B, T, N]

        dec = self.normalize_layers[0](dec, "denorm")
        return dec
