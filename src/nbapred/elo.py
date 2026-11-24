"""538-style Elo computed from game results, so the model works on any game log.

K=20, home-court advantage 100 points, margin-of-victory multiplier with the
autocorrelation correction, and 25% regression to the mean between seasons.
Pre-game ratings are recorded before each update, so they are safe model inputs.
Games with an unknown result (upcoming games) get a pre-game rating but no update.
"""
from __future__ import annotations

import numpy as np

MEAN = 1500.0


def expected(elo_diff):
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(elo_diff, dtype=float) / 400.0))


def run_elo(home_id, away_id, season, home_margin, *, k=20.0, hca=100.0, carry=0.75, init=1500.0):
    """Inputs are parallel arrays in chronological order. Returns (home_elo, away_elo) pre-game."""
    n = len(home_id)
    ratings: dict = {}
    last_season: dict = {}
    eh, ea = np.empty(n), np.empty(n)
    for i in range(n):
        for t in (home_id[i], away_id[i]):
            r = ratings.get(t, init)
            if t in last_season and last_season[t] != season[i]:
                r = carry * r + (1.0 - carry) * MEAN
            ratings[t] = r
            last_season[t] = season[i]
        h, a = ratings[home_id[i]], ratings[away_id[i]]
        eh[i], ea[i] = h, a
        m = home_margin[i]
        if np.isnan(m):
            continue  # upcoming game: no update
        diff = h + hca - a
        p = float(expected(diff))
        win = 1.0 if m > 0 else 0.0
        winner_diff = diff if m > 0 else -diff
        mult = ((abs(m) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = k * mult * (win - p)
        ratings[home_id[i]] = h + delta
        ratings[away_id[i]] = a - delta
    return eh, ea
