import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Embed import DataEmbedding


class _AnomalyAttention(nn.Module):
    """
    Anomaly Attention from Xu et al., ICLR 2022.

    Computes two attention distributions per head:
      - Series association S: standard scaled dot-product attention (learned)
      - Prior association P:  Gaussian kernel over position distances (learned sigma)

    The learned log_sigma per head controls how wide the "normal context window"
    is. For ECG at 100 Hz with seq_len=100, sigma will learn to match the
    typical QRS context width (~5-15 positions = 50-150 ms).

    The symmetric KL between P and S (association discrepancy) is returned so the
    caller can optionally add it to the training loss. Output uses series attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model)
        # One log-sigma per head — exponentiated so sigma is always positive.
        # Initialized to 0 → sigma=1; will grow to match the signal's context scale.
        self.log_sigma = nn.Parameter(torch.zeros(n_heads))
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        H, d    = self.n_heads, self.head_dim

        Q = self.Wq(x).view(B, L, H, d).transpose(1, 2)  # [B, H, L, d]
        K = self.Wk(x).view(B, L, H, d).transpose(1, 2)
        V = self.Wv(x).view(B, L, H, d).transpose(1, 2)

        # Series association: standard scaled dot-product attention [B, H, L, L]
        series = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        series = self.dropout(series)

        # Prior association: Gaussian kernel over pairwise position distances [B, H, L, L]
        pos    = torch.arange(L, dtype=x.dtype, device=x.device)
        dist_sq = (pos.unsqueeze(0) - pos.unsqueeze(1)) ** 2          # [L, L]
        sigma   = self.log_sigma.exp().clamp(min=0.1)                  # [H]
        prior   = F.softmax(
            -dist_sq.unsqueeze(0) / (2.0 * sigma.view(H, 1, 1) ** 2),
            dim=-1,
        ).unsqueeze(0).expand(B, -1, -1, -1)                           # [B, H, L, L]

        # Symmetric KL divergence, averaged over all batch/head/position entries
        eps   = 1e-8
        kl_ps = (prior  * (prior .add(eps).log() - series.add(eps).log())).sum(-1)
        kl_sp = (series * (series.add(eps).log() - prior .add(eps).log())).sum(-1)
        assoc_disc = (kl_ps + kl_sp).mean()

        out = torch.matmul(series, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.Wo(out), assoc_disc


class _AnomalyTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, activation: str = "gelu"):
        super().__init__()
        self.attn  = _AnomalyAttention(d_model, n_heads, dropout)
        act        = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), act, nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, disc = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, disc


class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer for reconstruction-based anomaly detection.

    Replaces standard self-attention with AnomalyAttention, which jointly
    computes a series association (learned softmax attention) and a prior
    association (learnable per-head Gaussian kernel). The symmetric KL
    between these two distributions — the association discrepancy — acts
    as an auxiliary signal: normal timesteps attend broadly (matching the
    Gaussian prior); anomalous timesteps concentrate attention on nearby
    points and deviate from the prior.

    Training:
        forward() returns x_hat and stores self.assoc_loss = assoc_lambda *
        total_discrepancy. The FL client picks this up and adds it to the
        MSE reconstruction loss. All other models in the registry don't have
        this attribute so the client's getattr fallback leaves them unchanged.

    Inference:
        self.assoc_loss is set to None during eval(); reconstruction MSE is
        the anomaly score (consistent with every other model in the registry).

    Reference: Xu et al., "Anomaly Transformer: Time Series Anomaly Detection
               with Association Discrepancy", ICLR 2022.
    """

    def __init__(self, configs):
        super().__init__()
        embed      = getattr(configs, "embed",        "fixed")
        freq       = getattr(configs, "freq",         "h")
        activation = getattr(configs, "activation",   "gelu")
        # Scales association discrepancy relative to reconstruction MSE.
        # 3e-4 contributes ~10-30% of total loss once the model warms up.
        self.assoc_lambda = getattr(configs, "assoc_lambda", 3e-4)
        self.assoc_loss   = None  # populated each forward; FL client reads this

        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout)
        self.layers = nn.ModuleList([
            _AnomalyTransformerLayer(
                configs.d_model, configs.n_heads, configs.d_ff,
                configs.dropout, activation,
            )
            for _ in range(configs.e_layers)
        ])
        self.norm       = nn.LayerNorm(configs.d_model)
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.embedding(x, None)

        total_disc = torch.zeros(1, device=x.device)
        for layer in self.layers:
            enc, disc = layer(enc)
            total_disc = total_disc + disc

        x_hat = self.projection(self.norm(enc))

        self.assoc_loss = self.assoc_lambda * total_disc if self.training else None

        return x_hat
