import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Embed import DataEmbedding


def parallel_scan(A, B):
    """
    Parallel prefix scan for h_t = A_t * h_{t-1} + B_t.

    Uses the associative operator (A2,b2)∘(A1,b1) = (A2*A1, A2*b1+b2) to
    do a tree-reduction over time: O(log L) depth instead of O(L) sequential.

    A: (B, L, D, 1)  scalar decay per channel per timestep
    B: (B, L, D, N)  input contribution per timestep
    Returns h: (B, L, D, N)
    """
    B_size, L, D, N = B.shape

    L_pad = 2 ** math.ceil(math.log2(L)) if L > 1 else 1
    if L_pad > L:
        pad = L_pad - L
        A = F.pad(A, (0, 0, 0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))

    # upsweep: build tree bottom-up
    A_tree = [A]
    B_tree = [B]
    length = L_pad
    while length > 1:
        length //= 2
        A_prev, B_prev = A_tree[-1], B_tree[-1]
        A_left,  A_right = A_prev[:, 0::2], A_prev[:, 1::2]
        B_left,  B_right = B_prev[:, 0::2], B_prev[:, 1::2]
        A_tree.append(A_right * A_left)
        B_tree.append(A_right * B_left + B_right)

    # downsweep: push prefixes back down
    h_tree = [torch.zeros_like(B_tree[-1])]
    for i in range(len(A_tree) - 1, 0, -1):
        h_parent = h_tree[-1]
        A_cur, B_cur = A_tree[i - 1], B_tree[i - 1]
        h_left  = A_cur[:, 0::2] * h_parent + B_cur[:, 0::2]
        h_right = A_cur[:, 1::2] * h_left   + B_cur[:, 1::2]
        length_cur = h_left.shape[1]
        h_full = torch.zeros(B_size, length_cur * 2, D, N, device=A.device, dtype=A.dtype)
        h_full[:, 0::2] = h_left
        h_full[:, 1::2] = h_right
        h_tree.append(h_full)

    return h_tree[-1][:, :L]


class Mamba2Block(nn.Module):
    """
    Mamba 2 block with grouped heads and scalar A decay.

    Differences from Mamba 1:
    - A is a learned scalar per (head, state) instead of a full matrix
    - Parallel scan replaces the sequential loop → O(log L) depth
    - Grouped heads (n_heads) over d_inner
    """
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, n_heads=4):
        super().__init__()
        self.d_inner  = int(expand * d_model)
        self.d_state  = d_state
        self.n_heads  = n_heads
        assert self.d_inner % n_heads == 0
        self.head_dim = self.d_inner // n_heads

        # z, x_branch, B, C, dt
        self.in_proj = nn.Linear(
            d_model, 2 * self.d_inner + 2 * d_state + n_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        # scalar decay: (n_heads, d_state), kept negative so A = -exp(A_log) < 0
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0).expand(n_heads, -1).clone()
        )
        self.D        = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        proj = self.in_proj(x)
        z, x_branch, B_mat, C_mat, dt = proj.split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.n_heads],
            dim=-1,
        )

        # causal depthwise conv
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :L].transpose(1, 2)
        x_branch = F.silu(x_branch)

        # discretise dt, expand from n_heads to d_inner
        dt = F.softplus(dt).repeat_interleave(self.head_dim, dim=-1)   # (B,L,d_inner)
        A  = -torch.exp(self.A_log).repeat_interleave(self.head_dim, dim=0)  # (d_inner,d_state)

        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B,L,d_inner,d_state)
        dB = dt.unsqueeze(-1) * B_mat.unsqueeze(2)                       # (B,L,d_inner,d_state)
        Bu = dB * x_branch.unsqueeze(-1)                                 # (B,L,d_inner,d_state)

        h = parallel_scan(dA, Bu)                                        # (B,L,d_inner,d_state)
        y = (h * C_mat.unsqueeze(2)).sum(dim=-1) + x_branch * self.D
        y = y * F.silu(z)

        return self.out_proj(y) + residual


class Mamba2(nn.Module):
    """
    Mamba2 for anomaly detection.

    Pure-PyTorch — no mamba_ssm / CUDA kernel required.
    Runs on Jetson Nano (GPU), Jetson Xavier (GPU), Raspberry Pi (CPU),
    and workstations (CPU/CUDA/MPS).
    """
    def __init__(self, configs):
        super().__init__()
        d_state  = getattr(configs, 'd_ff',    64)
        d_conv   = getattr(configs, 'd_conv',   4)
        expand   = getattr(configs, 'expand',   2)
        n_heads  = getattr(configs, 'n_heads',  4)
        e_layers = getattr(configs, 'e_layers', 1)
        embed    = getattr(configs, 'embed',   'fixed')
        freq     = getattr(configs, 'freq',    'h')

        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed, freq, configs.dropout
        )
        self.layers = nn.ModuleList([
            Mamba2Block(configs.d_model, d_state, d_conv, expand, n_heads)
            for _ in range(e_layers)
        ])
        self.norm      = nn.LayerNorm(configs.d_model)
        self.out_layer = nn.Linear(configs.d_model, configs.c_out, bias=False)

    def forward(self, x):
        x = self.embedding(x, None)
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(self.norm(x))
