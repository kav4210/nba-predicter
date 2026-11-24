"""Run:  pytest exercises -q     (unimplemented exercises show as skipped, not failed)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))


def graded(fn, *a, **k):
    try:
        return fn(*a, **k)
    except NotImplementedError:
        pytest.skip("not implemented yet")


def test_ex1_elo_update_matches_reference():
    from ex1_elo_update import elo_update
    from nbapred.elo import run_elo

    for margin in (12.0, -7.0, 1.0, -30.0):
        h, a = graded(elo_update, 1580.0, 1490.0, margin)
        eh, ea = run_elo(np.array([1, 1]), np.array([2, 2]), np.array([2020, 2020]), np.array([0.0, np.nan]))
        # reference: replay the single game from the same starting ratings
        from nbapred.elo import expected
        diff = 1580.0 + 100.0 - 1490.0
        p = float(expected(diff))
        win = 1.0 if margin > 0 else 0.0
        wd = diff if margin > 0 else -diff
        delta = 20.0 * (((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * wd)) * (win - p)
        assert h == pytest.approx(1580.0 + delta) and a == pytest.approx(1490.0 - delta)
        assert (h + a) == pytest.approx(1580.0 + 1490.0)  # zero-sum


def test_ex1_upset_moves_ratings_more_than_expected_win():
    from ex1_elo_update import elo_update
    fav_wins = graded(elo_update, 1700.0, 1400.0, 10.0)[0] - 1700.0
    upset = graded(elo_update, 1700.0, 1400.0, -10.0)[0] - 1700.0
    assert abs(upset) > abs(fav_wins)


def test_ex2_ipw_recovers_known_effect_and_beats_naive():
    from ex2_ipw import ipw_ate
    rng = np.random.default_rng(0)
    n = 40000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-1.2 * x))
    t = (rng.uniform(size=n) < p).astype(float)
    y = 2 * x - 1.5 * t + rng.normal(size=n)
    est = graded(ipw_ate, y, t, p)
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(est - (-1.5)) < 0.15 and abs(naive - (-1.5)) > 0.8


def test_ex3_attention_matches_torch_and_masks():
    from ex3_attention import attention
    torch.manual_seed(0)
    Q, K, V = torch.randn(3, 4, 8), torch.randn(3, 6, 8), torch.randn(3, 6, 5)
    out, w = graded(attention, Q, K, V)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 4, 5) and w.shape == (3, 4, 6)
    assert torch.allclose(w.sum(-1), torch.ones(3, 4), atol=1e-5)
    assert torch.allclose(out, ref, atol=1e-5)
    mask = torch.ones(3, 4, 6, dtype=torch.bool)
    mask[..., 4:] = False
    _, wm = attention(Q, K, V, mask)
    assert torch.all(wm[..., 4:] == 0) and torch.allclose(wm.sum(-1), torch.ones(3, 4), atol=1e-5)


def test_ex4_temperature_matches_reference_and_improves_log_loss():
    from ex4_temperature import fit_temperature
    from nbapred.train import fit_temperature as ref
    rng = np.random.default_rng(1)
    z = rng.normal(scale=3.0, size=5000)          # overconfident logits
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-z / 2.5))).astype(float)
    T = graded(fit_temperature, z, y)
    assert T == pytest.approx(ref(z, y.astype(np.float32)), rel=0.05)
    assert 2.0 < T < 3.2


def test_ex5_leak_is_fixed():
    from ex5_find_the_leak import rolling_win_pct
    rng = np.random.default_rng(0)
    rows = [(t, pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), int(rng.integers(0, 2)))
            for t in ("A", "B") for i in range(40)]
    df = pd.DataFrame(rows, columns=["team", "date", "win"])
    base = rolling_win_pct(df)
    flipped = df.copy()
    flipped.loc[df.index[25], "win"] = 1 - df.loc[df.index[25], "win"]   # change team A's game #25 result
    after = rolling_win_pct(flipped)
    assert base.loc[df.index[25]] == pytest.approx(after.loc[df.index[25]], nan_ok=True), \
        "the feature for a game moved when that game's own result changed: it still leaks"
    assert base.loc[df.index[24]] == pytest.approx(after.loc[df.index[24]], nan_ok=True)
    # and the fix must not destroy the signal: a later game should still depend on game #25
    assert not np.isclose(base.loc[df.index[26]], after.loc[df.index[26]])
