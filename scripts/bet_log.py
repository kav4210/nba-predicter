"""Paper-trade log: model probability vs Kalshi price, settled from real results.

  python scripts/bet_log.py add 2026-11-25 CLE@NYK NYK 0.595 0.666 --note "Mobley out"
  python scripts/bet_log.py settle      # fills results from data/modern_games.csv (run update_data.py first)
  python scripts/bet_log.py report      # does the model's gap vs the market predict anything?

One row = one side of one game (the side's win probability). The log file (paper_log.csv) is personal
and git-ignored. The question it answers: when the model disagrees with the market, who is right?
Brier of model vs market on the same games is the test; hit rate alone is not.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = ROOT / "paper_log.csv"
COLS = ["date", "game", "side", "price", "model_p", "note", "won"]


def load() -> pd.DataFrame:
    return pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)


def add(a):
    df = load()
    df.loc[len(df)] = [a.date, a.game.upper(), a.side.upper(), a.price, a.model_p, a.note or "", None]
    df.to_csv(LOG, index=False)
    print(f"logged {a.game} {a.side} price {a.price} model {a.model_p}")


def settle(_):
    from nbapred import modern
    g = modern.games_from_team_rows(modern.load_team_rows())
    g["date"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    res = {(r.date, f"{r.away_abbr}@{r.home_abbr}"): int(r.home_pts > r.away_pts) for r in g.itertuples()}
    df = load()
    n = 0
    for i, r in df[df["won"].isna()].iterrows():
        k = (str(r["date"]), r["game"])
        if k in res:
            home = r["game"].split("@")[1]
            df.loc[i, "won"] = res[k] if r["side"] == home else 1 - res[k]
            n += 1
    df.to_csv(LOG, index=False)
    print(f"settled {n}; {int(df['won'].isna().sum())} still open")


def report(_):
    df = load().dropna(subset=["won"])
    if df.empty:
        sys.exit("nothing settled yet")
    df["won"] = df["won"].astype(float)
    df["gap"] = df["model_p"] - df["price"]
    bm = ((df["model_p"] - df["won"]) ** 2).mean()
    bk = ((df["price"] - df["won"]) ** 2).mean()
    print(f"n={len(df)}  Brier model {bm:.4f}  market {bk:.4f}  (lower wins; need ~300+ games to tell them apart)")
    big = df[df["gap"].abs() >= 0.05]
    if len(big):
        side_model = (big["gap"] > 0)
        right = ((side_model & (big["won"] == 1)) | (~side_model & (big["won"] == 0))).mean()
        print(f"gaps >= 5 pts: n={len(big)}, model's side won {right:.0%}")
    print(f"mean gap {df['gap'].mean():+.3f}, mean realized minus price {(df['won'] - df['price']).mean():+.3f}")


p = argparse.ArgumentParser()
s = p.add_subparsers(dest="cmd", required=True)
x = s.add_parser("add")
x.add_argument("date"); x.add_argument("game"); x.add_argument("side")
x.add_argument("price", type=float); x.add_argument("model_p", type=float); x.add_argument("--note")
x.set_defaults(f=add)
s.add_parser("settle").set_defaults(f=settle)
s.add_parser("report").set_defaults(f=report)
a = p.parse_args()
a.f(a)
