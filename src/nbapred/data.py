"""Data loading. Source: FiveThirtyEight's public NBA Elo dataset (1946-2015)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "nbaallelo.csv"


def download(path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


def load_raw(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    return pd.read_csv(download(Path(path)))


def prepare_long(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per team per regular-season NBA game, sorted by franchise then date.

    The source has two rows per game (one per team's perspective). Playoffs and the
    ABA are dropped. Sorting by (franchise, date) makes each franchise's history a
    contiguous block, which the feature and sequence builders rely on.
    """
    df = raw[(raw.is_playoffs == 0) & (raw.lg_id == "NBA")].copy()
    df["date"] = pd.to_datetime(df["date_game"], format="%m/%d/%Y")
    df["margin"] = (df["pts"] - df["opp_pts"]).astype(float)
    df["win"] = (df["game_result"] == "W").astype(float)
    df["is_home"] = (df["game_location"] == "H").astype(float)
    df["is_neutral"] = (df["game_location"] == "N").astype(float)
    df = df.sort_values(["fran_id", "date", "gameorder"], kind="stable").reset_index(drop=True)
    df["rowid"] = range(len(df))
    return df
