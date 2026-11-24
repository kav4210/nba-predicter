"""Refresh data/modern_games.csv with the latest completed games from the NBA stats API.

Run this on YOUR machine (stats.nba.com blocks most cloud IPs and rejects some networks):

    pip install nba_api
    python scripts/update_data.py

It starts from the public 2010-2024 history, re-fetches the last season already in the file
through the current season, and rewrites the CSV. Then run scripts/train_production.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import modern  # noqa: E402

NEEDED = ["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "PTS", "PLUS_MINUS"]


def season_string(end_year: int) -> str:
    return f"{end_year - 1}-{end_year % 100:02d}"


def normalize(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """nba_api LeagueGameLog (team mode) frame -> the columns the pipeline stores."""
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"nba_api response is missing columns {missing}; got {list(df.columns)}")
    out = df.dropna(subset=["WL"]).copy()  # games still in progress have no result yet
    out["SEASON_YEAR"] = season
    out["GAME_ID"] = out["GAME_ID"].astype(str)
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.strftime("%Y-%m-%dT00:00:00")
    return out[modern.KEEP]


def fetch_season(season: str, retries: int = 3) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    for attempt in range(1, retries + 1):
        try:
            log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="T",
                                              season_type_all_star="Regular Season", timeout=60)
            return normalize(log.get_data_frames()[0], season)
        except Exception as e:  # network errors, rate limits
            print(f"  attempt {attempt} failed for {season}: {e}")
            time.sleep(3 * attempt)
    raise SystemExit(f"Could not fetch {season}. stats.nba.com often blocks cloud and VPN IPs; run this at home.")


def main():
    path = modern.download_history()
    have = pd.read_csv(path, dtype={"GAME_ID": str})
    last = pd.to_datetime(have["GAME_DATE"]).max()
    today = pd.Timestamp.today()
    first_end = int(modern.season_end_year(pd.Series([last])).iloc[0])
    cur_end = int(modern.season_end_year(pd.Series([today])).iloc[0])
    seasons = [season_string(y) for y in range(first_end, cur_end + 1)]
    print(f"history ends {last.date()}; fetching {seasons}")

    fresh = []
    for s in seasons:
        df = fetch_season(s)
        print(f"  {s}: {len(df) // 2} completed games")
        fresh.append(df)
        time.sleep(1.5)
    keep = have[~have["SEASON_YEAR"].isin(seasons)]
    out = pd.concat([keep, *fresh], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"wrote {path} ({len(out) // 2} games, latest {pd.to_datetime(out['GAME_DATE']).max().date()})")


if __name__ == "__main__":
    main()
