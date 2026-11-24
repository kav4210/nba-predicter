"""Leakage tests: features for a game must not change if the future is rewritten."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbapred import features as F  # noqa: E402
from nbapred.data import load_raw, prepare_long  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def build(raw):
    long_feat = F.add_team_features(prepare_long(raw))
    return long_feat, F.build_games(long_feat)


def test_forbidden_columns_never_used():
    for use_elo in (False, True):
        assert not set(F.feature_columns(use_elo)) & F.FORBIDDEN_INPUTS
        assert not any("forecast" in c or "elo_n" in c for c in F.feature_columns(use_elo))


def test_no_elo_variant_has_no_elo_derived_feature():
    cols = F.feature_columns(use_elo=False)
    assert not any("elo" in c or "sos" in c for c in cols)


def _tamper(raw, mask, seed=0):
    t = raw.copy()
    rng = np.random.default_rng(seed)
    for col in ("pts", "opp_pts"):
        t.loc[mask, col] = rng.integers(60, 140, mask.sum())
    t.loc[mask, "game_result"] = rng.choice(["W", "L"], mask.sum())
    return t


def test_features_ignore_own_and_future_outcomes(raw):
    """Rewrite results of every game on/after a cutoff day (including that day's own games).

    Features for all games up to and including the cutoff day must not move. Rewriting
    only the strict future would miss a feature that peeks at its own game's result.
    """
    dates = pd.to_datetime(raw["date_game"], format="%m/%d/%Y")
    cutoff = pd.Timestamp("2005-02-01")
    assert (dates == cutoff).sum() >= 4  # make sure the cutoff day actually has games
    tampered = _tamper(raw, (dates >= cutoff).to_numpy())

    base_long, base_games = build(raw)
    new_long, new_games = build(tampered)

    upto = base_games[base_games.date <= cutoff].set_index("game_id")
    upto_new = new_games[new_games.date <= cutoff].set_index("game_id").loc[upto.index]
    cols = F.feature_columns(True)
    pd.testing.assert_frame_equal(upto[cols], upto_new[cols], check_exact=False, rtol=1e-9, atol=1e-9)

    ok = F.eligible(upto)
    s0 = F.build_sequences(base_long, upto.loc[ok, "home_rowid"].to_numpy(), use_elo=True)
    s1 = F.build_sequences(new_long, upto_new.loc[ok, "home_rowid"].to_numpy(), use_elo=True)
    np.testing.assert_allclose(s0, s1)

    # sanity: the tampering really did change outcomes, so the test has teeth
    cut_day = new_games[new_games.date == cutoff].set_index("game_id").loc[
        base_games[base_games.date == cutoff].game_id]
    assert (cut_day["y"].to_numpy() != base_games[base_games.date == cutoff]["y"].to_numpy()).any()


def test_sequence_contains_only_prior_games(raw):
    long_feat, games = build(raw)
    g = games[F.eligible(games)].iloc[5000:5200]
    rid = g["home_rowid"].to_numpy()
    seq = F.build_sequences(long_feat, rid, use_elo=True)
    # days_ago column must be strictly positive: every step is earlier than the current game
    days_ago = seq[:, :, -2]
    assert (days_ago > 0).all()
    # and all K steps come from the same franchise
    idx = rid[:, None] + np.arange(-F.K, 0)[None, :]
    fr = long_feat["fran_id"].to_numpy()
    assert (fr[idx] == fr[rid][:, None]).all()
