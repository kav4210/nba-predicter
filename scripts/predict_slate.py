"""Predict the winners of one day's games.

    python scripts/predict_slate.py --date 2026-11-25                       # schedule from the NBA's public JSON
    python scripts/predict_slate.py --date 2026-11-25 --games "BOS@LAL,NYK@MIA"   # manual, AWAY@HOME

Requires data/modern_games.csv to be fresh (python scripts/update_data.py) and a trained model
(python scripts/train_production.py). If the data is more than 3 days older than the slate
date the script refuses to run, because stale Elo and form give confident nonsense.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import modern  # noqa: E402
from nbapred.predict import load_bundle, predict_home_win, slate_features  # noqa: E402

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
MAX_STALE_DAYS = 3


def parse_schedule(payload: dict, date: pd.Timestamp) -> list[tuple[str, str]]:
    """(away, home) tricodes for regular-season games on `date` from the NBA schedule JSON."""
    day = f"{date:%Y-%m-%d}"
    out = []
    for gd in payload["leagueSchedule"]["gameDates"]:
        for g in gd["games"]:
            when = str(g.get("gameDateEst") or g.get("gameDateTimeEst") or "")[:10]
            if when == day and str(g["gameId"]).startswith("002"):  # 002 = regular season
                out.append((g["awayTeam"]["teamTricode"], g["homeTeam"]["teamTricode"]))
    return out


def fetch_matchups(date: pd.Timestamp) -> list[tuple[str, str]]:
    with urllib.request.urlopen(SCHEDULE_URL, timeout=30) as r:
        return parse_schedule(json.load(r), date)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--games", help='manual slate, AWAY@HOME comma-separated, e.g. "BOS@LAL,NYK@MIA"')
    ap.add_argument("--data", default=str(modern.DEFAULT_PATH))
    ap.add_argument("--allow-stale", action="store_true")
    args = ap.parse_args()
    date = pd.Timestamp(args.date).normalize()

    completed = modern.games_from_team_rows(modern.load_team_rows(Path(args.data)))
    completed = completed[completed["date"] < date]  # never use results from the slate date or later
    last = completed["date"].max()
    gap = (date - last).days
    if gap > MAX_STALE_DAYS and not args.allow_stale:
        sys.exit(f"Data ends {last.date()}, {gap} days before {date.date()}. Run scripts/update_data.py first "
                 f"(or pass --allow-stale to see what stale data produces).")

    if args.games:
        matchups = [tuple(x.strip().upper().split("@")) for x in args.games.split(",")]
    else:
        matchups = fetch_matchups(date)
    if not matchups:
        sys.exit(f"No regular-season games found for {date.date()}.")

    upcoming = modern.make_upcoming(completed, matchups, date)
    feats = slate_features(completed, upcoming)
    preds = predict_home_win(load_bundle(), feats)

    out = pd.DataFrame({
        "away": feats["away_team"], "home": feats["home_team"],
        "p_home_win": preds["p_home"].round(3),
        "pick": [h if p >= 0.5 else a for h, a, p in zip(feats["home_team"], feats["away_team"], preds["p_home"])],
        "confidence": (preds["p_home"].sub(0.5).abs().add(0.5)).round(3),
        "elo_gap_home": feats["d_elo"].round(0),
        "away_rest_days": feats["a_rest_days"], "home_rest_days": feats["h_rest_days"],
    }).sort_values("confidence", ascending=False)
    print(f"\nPredictions for {date.date()} (data through {last.date()})\n")
    print(out.to_string(index=False))
    (ROOT / "predictions").mkdir(exist_ok=True)
    out.to_csv(ROOT / "predictions" / f"{date:%Y-%m-%d}.csv", index=False)


if __name__ == "__main__":
    main()
