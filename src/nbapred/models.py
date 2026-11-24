"""PyTorch models. Every model returns a raw logit for P(home team wins)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SeqNet(nn.Module):
    """Shared GRU reads each team's last K games; its summary is fused with tabular features.

    Idea: the tabular model sees fixed summaries (last-10 average). The GRU can learn
    its own weighting of recent games (recency, home/away context, opponent strength).
    Symmetric design: the same encoder runs on home and away, and the head sees
    [h_home, h_away, h_home - h_away, tabular].
    """

    def __init__(self, step_dim: int, tab_dim: int, gru_hidden: int = 32, head=(64,), dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(step_dim, gru_hidden, batch_first=True)
        self.tab = nn.Sequential(nn.Linear(tab_dim, 64), nn.GELU(), nn.Dropout(dropout))
        d = gru_hidden * 3 + 64
        layers = []
        for h in head:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def encode(self, seq):
        _, h = self.gru(seq)
        return h[-1]

    def forward(self, seq_home, seq_away, x_tab):
        hh, ha = self.encode(seq_home), self.encode(seq_away)
        z = torch.cat([hh, ha, hh - ha, self.tab(x_tab)], dim=1)
        return self.head(z).squeeze(-1)


class _Block(nn.Module):
    """Pre-norm transformer block written out so the attention weights can be inspected."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self.keep_attn = False
        self.last_attn = None

    def forward(self, x):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, need_weights=self.keep_attn, average_attn_weights=True)
        if self.keep_attn:
            self.last_attn = w.detach()
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.ln2(x)))


class TransNet(nn.Module):
    """Transformer encoder over each team's last K games (self-attention instead of a GRU).

    A learned [CLS] token is prepended to the K game tokens; its final state summarises the team.
    Learned positional embeddings tell the model which game is most recent. Same fusion head as
    SeqNet, so the two differ only in how a history is encoded.
    """

    def __init__(self, step_dim: int, tab_dim: int, k: int = 10, d_model: int = 32, heads: int = 4,
                 layers: int = 2, ff: int = 64, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(step_dim, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, k + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([_Block(d_model, heads, ff, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.tab = nn.Sequential(nn.Linear(tab_dim, 64), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(d_model * 3 + 64, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def encode(self, seq):
        x = self.inp(seq)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.ln_f(x[:, 0])

    def forward(self, seq_home, seq_away, x_tab):
        hh, ha = self.encode(seq_home), self.encode(seq_away)
        return self.head(torch.cat([hh, ha, hh - ha, self.tab(x_tab)], dim=1)).squeeze(-1)

    @torch.no_grad()
    def cls_attention(self, seq):
        """Attention from [CLS] to each of the K games in the last layer, averaged over heads: (B, K)."""
        self.eval()
        last = self.blocks[-1]
        last.keep_attn = True
        self.encode(seq)
        last.keep_attn = False
        return last.last_attn[:, 0, 1:]
