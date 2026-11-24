"""Exercise 5: find and fix the leak.

`rolling_win_pct` is supposed to give each team-game its win percentage over the team's previous
10 games, using only information available before that game. It has a leak. Fix it in place.

The grader changes the result of the game being predicted and checks that its feature does not
move. Before fixing, write down in LEARN.md (question 1) exactly which line leaks and what a model
trained on this feature would learn to do.
"""
import pandas as pd


def rolling_win_pct(df: pd.DataFrame) -> pd.Series:
    """df columns: team (str), date (datetime), win (0/1). Returns a Series aligned with df.index."""
    df = df.sort_values(["team", "date"])
    out = df.groupby("team")["win"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    return out.reindex(df.index)
