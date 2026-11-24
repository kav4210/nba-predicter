"""The deployment path (upcoming games, results unknown) must match the backtest path exactly."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import features as F  # noqa: E402
from nbapred import modern  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402
from nbapred.elo import expected, run_elo  # noqa: E402
from nbapred.predict import slate_features  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def completed():
    return modern.games_from_team_rows(modern.load_team_rows())


@pytest.mark.parametrize("day", ["2019-01-15", "2023-02-01", "2024-03-20"])
def test_slate_features_equal_backtest_features(completed, day):
    day = pd.Timestamp(day)
    todays = completed[completed["date"] == day]
    assert len(todays) >= 3

    # backtest path: everything is known, features computed in one pass
    batch = F.build_games(F.add_team_features(prepare_long(modern.to_long(completed))))
    batch = batch.set_index("game_id").loc[todays["game_id"]]

    # deployment path: only games before `day` are known; today's games are upcoming
    history = completed[completed["date"] < day]
    matchups = [(r.away_abbr, r.home_abbr) for r in todays.itertuples()]
    upcoming = modern.make_upcoming(history, matchups, day)
    feats = slate_features(history, upcoming)

    cols = F.feature_columns(True)
    key = lambda df: df.assign(k=list(zip(df["home_team"], df["away_team"]))).set_index("k")[cols].sort_index()  # noqa: E731
    got = key(feats)
    want = key(batch.reset_index())
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9, atol=1e-9)


def test_upcoming_results_never_change_features(completed):
    """Whatever placeholder scores an upcoming game carries, its own features must not move."""
    day = pd.Timestamp("2023-02-01")
    history = completed[completed["date"] < day]
    todays = completed[completed["date"] == day]
    up = modern.make_upcoming(history, [(r.away_abbr, r.home_abbr) for r in todays.itertuples()], day)
    base = slate_features(history, up)
    up2 = up.copy()
    up2["home_pts"], up2["away_pts"] = 150.0, 60.0
    other = slate_features(history, up2)
    cols = F.feature_columns(True)
    pd.testing.assert_frame_equal(base[cols], other[cols])


def test_slate_rejects_repeated_team_and_unknown_abbreviation(completed):
    day = pd.Timestamp("2023-02-01")
    history = completed[completed["date"] < day]
    with pytest.raises(ValueError, match="twice"):
        modern.make_upcoming(history, [("BOS", "LAL"), ("BOS", "NYK")], day)
    with pytest.raises(ValueError, match="unknown team"):
        modern.make_upcoming(history, [("XXX", "LAL")], day)


def test_elo_is_zero_sum_and_orders_sensibly():
    # 3 games; both teams start at 1500. A home win must raise home and lower away by the same amount.
    eh, ea = run_elo(np.array([1, 1, 2]), np.array([2, 2, 1]), np.array([2020, 2020, 2020]),
                     np.array([10.0, 10.0, np.nan]))
    assert eh[0] == ea[0] == 1500
    assert eh[1] > 1500 > ea[1]
    assert abs((eh[1] + ea[1]) - 3000) < 1e-9
    assert ea[2] > 1500 > eh[2]  # team 1 won twice, so as the game-3 away side it is the higher-rated team
    assert expected(0) == 0.5


def test_update_data_normalizes_nba_api_frame_and_drops_unfinished_games():
    upd = _load_script("update_data")
    df = pd.DataFrame({
        "TEAM_ID": [1, 2, 3, 4], "TEAM_ABBREVIATION": ["BOS", "LAL", "NYK", "MIA"],
        "GAME_ID": ["0022600001", "0022600001", "0022600002", "0022600002"],
        "GAME_DATE": ["2026-10-22"] * 4, "MATCHUP": ["BOS vs. LAL", "LAL @ BOS", "NYK vs. MIA", "MIA @ NYK"],
        "WL": ["W", "L", None, None], "PTS": [110, 100, 0, 0], "PLUS_MINUS": [10.0, -10.0, 0.0, 0.0],
    })
    out = upd.normalize(df, "2026-27")
    assert len(out) == 2 and list(out.columns) == modern.KEEP
    g = modern.games_from_team_rows(out)
    assert len(g) == 1 and g.iloc[0].home_abbr == "BOS" and g.iloc[0].home_pts == 110 and g.iloc[0].away_pts == 100
    with pytest.raises(ValueError, match="missing columns"):
        upd.normalize(df.drop(columns=["PLUS_MINUS"]), "2026-27")


def test_schedule_parser_keeps_only_regular_season_games_on_the_date():
    ps = _load_script("predict_slate")
    payload = {"leagueSchedule": {"gameDates": [{"games": [
        {"gameId": "0022600010", "gameDateEst": "2026-11-25T00:00:00Z",
         "homeTeam": {"teamTricode": "LAL"}, "awayTeam": {"teamTricode": "BOS"}},
        {"gameId": "0012600099", "gameDateEst": "2026-11-25T00:00:00Z",  # preseason id, must be skipped
         "homeTeam": {"teamTricode": "NYK"}, "awayTeam": {"teamTricode": "MIA"}},
        {"gameId": "0022600011", "gameDateEst": "2026-11-26T00:00:00Z",  # other day
         "homeTeam": {"teamTricode": "DEN"}, "awayTeam": {"teamTricode": "PHX"}},
    ]}]}}
    assert ps.parse_schedule(payload, pd.Timestamp("2026-11-25")) == [("BOS", "LAL")]
