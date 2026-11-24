"""Modern-era data: nba_api / stats.nba.com team game logs -> the schema the feature code expects.

Input rows (one per team per game) need: SEASON_YEAR, TEAM_ID, TEAM_ABBREVIATION, GAME_ID,
GAME_DATE, MATCHUP ('BOS vs. LAL' = BOS home, 'BOS @ LAL' = BOS away), WL, PTS, PLUS_MINUS.
That is what nba_api's LeagueGameLog (team mode) returns, and what the public
NocturneBear/NBA-Data-2010-2024 totals file contains.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .elo import expected, run_elo

URL = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/regular_season_totals_2010_2024.csv"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "modern_games.csv"
KEEP = ["SEASON_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS", "PLUS_MINUS"]


def download_history(path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(URL, tmp)
        pd.read_csv(tmp, usecols=KEEP).to_csv(path, index=False)
        tmp.unlink()
    return path


def load_team_rows(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(download_history(Path(path)), usecols=KEEP, dtype={"GAME_ID": str})


def season_end_year(dates: pd.Series) -> pd.Series:
    """Season label as the ending year (2023-24 -> 2024); a season starts in October."""
    return dates.dt.year + (dates.dt.month >= 9).astype(int)


def games_from_team_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per completed game, home vs away."""
    d = rows.dropna(subset=["WL"]).copy()
    d["date"] = pd.to_datetime(d["GAME_DATE"]).dt.normalize()
    d["opp_pts"] = d["PTS"] - d["PLUS_MINUS"]
    is_home = d["MATCHUP"].str.contains("vs.", regex=False)
    cols = {"TEAM_ID": "id", "TEAM_ABBREVIATION": "abbr", "PTS": "pts", "opp_pts": "opp"}
    h = d[is_home][["GAME_ID", "date", *cols]].rename(columns=cols)
    a = d[~is_home][["GAME_ID", *cols]].rename(columns=cols)
    g = h.merge(a, on="GAME_ID", suffixes=("_h", "_a"))
    out = pd.DataFrame({
        "game_id": g["GAME_ID"], "date": g["date"],
        "home_id": g["id_h"].astype(int), "away_id": g["id_a"].astype(int),
        "home_abbr": g["abbr_h"], "away_abbr": g["abbr_a"],
        "home_pts": g["pts_h"].astype(float), "away_pts": g["pts_a"].astype(float),
    })
    out["year_id"] = season_end_year(out["date"])
    out["is_upcoming"] = False
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def make_upcoming(completed: pd.DataFrame, matchups: list[tuple[str, str]], date) -> pd.DataFrame:
    """matchups: [(away_abbr, home_abbr), ...] on one date, resolved with abbreviations seen in the data."""
    latest = completed.sort_values("date").groupby("home_id").tail(1)
    abbr_to_id = {r.home_abbr: r.home_id for r in latest.itertuples()}
    for r in completed.sort_values("date").groupby("away_id").tail(1).itertuples():
        abbr_to_id.setdefault(r.away_abbr, r.away_id)
    rows = []
    for i, (away, home) in enumerate(matchups):
        for ab in (away, home):
            if ab not in abbr_to_id:
                raise ValueError(f"unknown team abbreviation {ab!r}; known: {sorted(abbr_to_id)}")
        rows.append({
            "game_id": f"UP{pd.Timestamp(date):%Y%m%d}{i:02d}", "date": pd.Timestamp(date).normalize(),
            "home_id": abbr_to_id[home], "away_id": abbr_to_id[away], "home_abbr": home, "away_abbr": away,
            "home_pts": np.nan, "away_pts": np.nan, "is_upcoming": True,
        })
    up = pd.DataFrame(rows)
    teams = pd.concat([up.home_id, up.away_id])
    if teams.duplicated().any():
        raise ValueError("a team appears twice in the slate; predict one date at a time")
    up["year_id"] = season_end_year(up["date"])
    return up


def to_long(games: pd.DataFrame) -> pd.DataFrame:
    """Games (completed + upcoming) -> two team rows per game in the FiveThirtyEight-style schema.

    Elo is computed here, in order, so `elo_i` is always pre-game. Upcoming games get placeholder
    scores (100-100, margin 0): features never use a game's own result (tests/test_no_leakage.py),
    so the placeholder cannot affect that game's features, and each team may appear only once in
    the upcoming slate.
    """
    g = games.sort_values(["date", "is_upcoming", "game_id"]).reset_index(drop=True)
    margin = (g["home_pts"] - g["away_pts"]).to_numpy(dtype=float)
    eh, ea = run_elo(g["home_id"].to_numpy(), g["away_id"].to_numpy(), g["year_id"].to_numpy(), margin)
    p_home = expected(eh + 100.0 - ea)
    up = g["is_upcoming"].to_numpy()
    hp = np.where(up, 100.0, g["home_pts"])
    ap = np.where(up, 100.0, g["away_pts"])
    date_str = g["date"].dt.strftime("%m/%d/%Y")

    def side(home: bool) -> pd.DataFrame:
        t, o = ("home", "away") if home else ("away", "home")
        won = (hp > ap) if home else (ap > hp)
        return pd.DataFrame({
            "gameorder": np.arange(1, len(g) + 1), "game_id": g["game_id"], "lg_id": "NBA",
            "_iscopy": 0 if home else 1, "year_id": g["year_id"], "date_game": date_str,
            "is_playoffs": 0, "team_id": g[f"{t}_abbr"], "fran_id": g[f"{t}_id"].astype(str),
            "pts": hp if home else ap, "opp_pts": ap if home else hp,
            "elo_i": eh if home else ea, "opp_elo_i": ea if home else eh,
            "game_location": "H" if home else "A", "game_result": np.where(won, "W", "L"),
            "forecast": p_home if home else 1 - p_home, "is_upcoming": up,
        })

    return pd.concat([side(True), side(False)], ignore_index=True)
