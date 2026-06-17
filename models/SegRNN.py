import torch
import torch.nn as nn


class SegRNN(nn.Module):
    """
    Segment-level GRU for anomaly detection.

    The input sequence is divided into non-overlapping segments, each
    embedded into d_model space. A GRU encodes the segment sequence into
    a hidden state. The decoder uses learned positional + channel embeddings
    as queries, running one GRU step per output segment, then projecting
    back to the original segment length.

    For anomaly detection pred_len == seq_len (full reconstruction).
    Edge-friendly: GRU is compute-light and causal, suitable for streaming.

    Source: tslib/models/SegRNN.py — stripped to anomaly_detection task;
    removed unused series_decomp import.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in  = configs.enc_in
        self.d_model = configs.d_model
        seg_len      = getattr(configs, "seg_len", 25)

        # pred_len == seq_len for reconstruction tasks
        pred_len = configs.seq_len
        assert configs.seq_len % seg_len == 0, (
            f"seq_len ({configs.seq_len}) must be divisible by seg_len ({seg_len})"
        )
        self.seg_len   = seg_len
        self.seg_num_x = configs.seq_len // seg_len
        self.seg_num_y = pred_len // seg_len

        self.value_embedding = nn.Sequential(
            nn.Linear(seg_len, configs.d_model),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            input_size=configs.d_model,
            hidden_size=configs.d_model,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.pos_emb     = nn.Parameter(torch.randn(self.seg_num_y, configs.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(configs.enc_in, configs.d_model // 2))
        self.predict = nn.Sequential(
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, seg_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        C = self.enc_in

        # Subtract last value for instance normalization
        seq_last = x[:, -1:, :].detach()          # [B, 1, C]
        x = (x - seq_last).permute(0, 2, 1)       # [B, C, T]

        # Segment + embed
        x = self.value_embedding(
            x.reshape(-1, self.seg_num_x, self.seg_len)
        )                                          # [B*C, seg_num_x, d_model]

        # Encode: hidden state summarises the full sequence
        _, hn = self.rnn(x)                        # hn: [1, B*C, d_model]

        # Build decoder queries: position × channel embeddings
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).expand(C, -1, -1),              # [C, seg_num_y, d//2]
            self.channel_emb.unsqueeze(1).expand(-1, self.seg_num_y, -1),  # [C, seg_num_y, d//2]
        ], dim=-1)                                                     # [C, seg_num_y, d_model]
        pos_emb = pos_emb.reshape(-1, 1, self.d_model).repeat(B, 1, 1)
        # [C*seg_num_y, 1, d_model] → repeat B times → [B*C*seg_num_y, 1, d_model]

        # Expand encoder hidden to cover all segment decode steps
        h0 = hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)
        # hn [1,B*C,d] → repeat → [1,B*C,d*seg_num_y] → view → [1,B*C*seg_num_y,d]

        _, hy = self.rnn(pos_emb, h0)              # hy: [1, B*C*seg_num_y, d_model]

        # Project each segment hidden → segment values
        y = self.predict(hy)                       # [1, B*C*seg_num_y, seg_len]
        y = y.view(-1, C, self.seq_len)            # [B, C, seq_len]
        y = y.permute(0, 2, 1) + seq_last          # [B, T, C] with de-normalization

        return y
