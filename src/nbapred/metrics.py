from __future__ import annotations

import numpy as np

EPS = 1e-7


def clip(p):
    return np.clip(p, EPS, 1 - EPS)


def per_game_logloss(p, y):
    p = clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(p, y):
    return float(per_game_logloss(p, y).mean())


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def accuracy(p, y):
    return float(np.mean((p >= 0.5) == (y == 1)))


def reliability(p, y, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    ids = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = ids == b
        if m.sum():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def ece(p, y, bins: int = 10):
    rows = reliability(p, y, bins)
    n = sum(r[2] for r in rows)
    return float(sum(abs(c - a) * k for c, a, k in rows) / n)


def summarize(p, y):
    return {"log_loss": log_loss(p, y), "brier": brier(p, y), "accuracy": accuracy(p, y), "ece": ece(p, y)}


def paired_bootstrap_delta(p_model, p_ref, y, n_boot: int = 2000, seed: int = 0):
    """95% CI for mean per-game log-loss difference (model - reference). Negative = model better."""
    d = per_game_logloss(p_model, y) - per_game_logloss(p_ref, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
