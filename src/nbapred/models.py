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
