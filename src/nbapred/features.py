"""Leakage-safe feature engineering.

Rule: every feature for a game may only use information available before tipoff.
  * Rolling/season stats use shift(1), so the current game is never included.
  * `elo_i` is the pre-game Elo rating (safe). `elo_n`, `win_equiv`, `forecast`,
    `pts`, `opp_pts`, `game_result` are post-game or outcome-derived and are never
    used as inputs. `forecast` is used only as an external benchmark.
tests/test_no_leakage.py enforces this by rewriting the future and checking that
past features do not move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 10  # games of history fed to the sequence model

FORBIDDEN_INPUTS = {
    "elo_n", "opp_elo_n", "win_equiv", "forecast", "pts", "opp_pts", "game_result",
    "win", "margin",
}

TEAM_FEATS = [
    "l10_wpct", "l10_margin", "l5_margin", "l10_sos",
    "season_wpct", "season_margin", "season_gp",
    "rest_days", "b2b", "games_last7",
]


def _count_prior_within(dates: np.ndarray, days: int) -> np.ndarray:
    d = dates.astype("datetime64[D]").astype(np.int64)
    idx = np.searchsorted(d, d - days, side="left")
    return np.arange(len(d)) - idx


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-game features for each team-game row. `df` must come from prepare_long."""
    out = df.copy()
    g = out.groupby("fran_id", sort=False)

    out["n_prior"] = g.cumcount()
    out["l10_wpct"] = g["win"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    out["l10_margin"] = g["margin"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    out["l5_margin"] = g["margin"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    out["l10_sos"] = g["opp_elo_i"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())

    gs = out.groupby(["fran_id", "year_id"], sort=False)
    out["season_gp"] = gs.cumcount()
    gp = out["season_gp"].replace(0, np.nan)
    out["season_wpct"] = ((gs["win"].cumsum() - out["win"]) / gp).fillna(0.5)
    out["season_margin"] = ((gs["margin"].cumsum() - out["margin"]) / gp).fillna(0.0)

    rest = g["date"].diff().dt.days
    out["rest_days"] = rest.fillna(10).clip(upper=10)
    out["b2b"] = (out["rest_days"] == 1).astype(float)
    out["games_last7"] = g["date"].transform(lambda s: pd.Series(_count_prior_within(s.values, 7), index=s.index))
    out["elo"] = out["elo_i"]
    return out


def build_games(long_feat: pd.DataFrame) -> pd.DataFrame:
    """One row per game, oriented home vs away (neutral-site: the source's primary row)."""
    home = long_feat[(long_feat.is_home == 1) | ((long_feat.is_neutral == 1) & (long_feat._iscopy == 0))]
    home = home.drop_duplicates("game_id")
    pair = long_feat.merge(home[["game_id", "fran_id"]].rename(columns={"fran_id": "home_fran"}), on="game_id")
    away = pair[pair.fran_id != pair.home_fran]
    assert len(home) == len(away) == long_feat.game_id.nunique(), "each game needs exactly one home and one away row"

    cols = TEAM_FEATS + ["elo"]
    h = home.set_index("game_id")
    a = away.set_index("game_id").loc[h.index]
    g = pd.DataFrame(index=h.index)
    g["date"] = h["date"]
    g["gameorder"] = h["gameorder"]
    g["year_id"] = h["year_id"]
    g["is_neutral"] = h["is_neutral"]
    g["home_rowid"] = h["rowid"]
    g["away_rowid"] = a["rowid"]
    g["h_n_prior"] = h["n_prior"]
    g["a_n_prior"] = a["n_prior"]
    for c in cols:
        g[f"h_{c}"] = h[c]
        g[f"a_{c}"] = a[c]
        g[f"d_{c}"] = h[c] - a[c]
    g["home_team"] = h["team_id"]
    g["away_team"] = a["team_id"]
    if "is_upcoming" in h.columns:
        g["is_upcoming"] = h["is_upcoming"].astype(bool)
    g["y"] = h["win"].astype(int)
    g["elo_forecast"] = h["forecast"]  # benchmark only, never an input
    g = g.sort_values(["date", "gameorder"]).reset_index()
    g["lg_home_rate"] = g["y"].rolling(1500, min_periods=200).mean().shift(1).fillna(0.6)
    # Games on the same date are simultaneous: all of them see the value as of the first game that day.
    g["lg_home_rate"] = g.groupby("date")["lg_home_rate"].transform("first")
    return g


def feature_columns(use_elo: bool) -> list[str]:
    # l10_sos is built from opponents' Elo, so the no-Elo ablation must drop it too.
    cols = [c for c in TEAM_FEATS if c != "l10_sos"] if not use_elo else TEAM_FEATS + ["elo"]
    f = [f"{p}_{c}" for c in cols for p in ("h", "a", "d")]
    return f + ["is_neutral", "lg_home_rate"]


def build_sequences(long_feat: pd.DataFrame, rowids: np.ndarray, use_elo: bool) -> np.ndarray:
    """Last K games for each requested team-game row -> array (n, K, F).

    Per-step features describe a *past* game from the team's point of view, plus two
    relative-to-now features (days since, same season). Requires the row to have at
    least K prior games for its franchise (enforced by the caller's filter).
    """
    lf = long_feat
    opp_diff = ((lf["opp_elo_i"] - lf["elo_i"]) / 200.0).to_numpy()
    step = [
        np.clip(lf["margin"].to_numpy() / 15.0, -3, 3),
        lf["win"].to_numpy(),
        lf["is_home"].to_numpy(),
        lf["rest_days"].to_numpy() / 10.0,
    ]
    if use_elo:
        step.append(opp_diff)
    S = np.stack(step, axis=1).astype(np.float32)
    dates = lf["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    years = lf["year_id"].to_numpy()

    idx = rowids[:, None] + np.arange(-K, 0)[None, :]
    assert (idx >= 0).all()
    base = S[idx]  # (n, K, F)
    days_ago = np.clip((dates[rowids][:, None] - dates[idx]) / 60.0, 0, 6)
    same_season = (years[idx] == years[rowids][:, None]).astype(np.float32)
    return np.concatenate([base, days_ago[..., None].astype(np.float32), same_season[..., None]], axis=2)


def eligible(games: pd.DataFrame) -> pd.Series:
    """Both teams need >= K prior games so sequences are fully populated."""
    return (games.h_n_prior >= K) & (games.a_n_prior >= K)
