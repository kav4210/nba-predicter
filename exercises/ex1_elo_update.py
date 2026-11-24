"""Exercise 1: one Elo update.

Implement elo_update so that it matches what src/nbapred/elo.py does for a single game.
Do not read elo.py until your test passes.

Rules (538 style):
  * expected home win prob = 1 / (1 + 10 ** (-(home + hca - away) / 400))
  * margin is home points minus away points (never 0 in the NBA)
  * margin-of-victory multiplier:  ((|margin| + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
      where winner_diff = (home + hca - away) if the home team won, else -(home + hca - away)
  * change = k * multiplier * (1 if home won else 0  -  expected home win prob)
  * home gains `change`, away loses the same amount
Return (new_home, new_away).
"""


def elo_update(home: float, away: float, margin: float, k: float = 20.0, hca: float = 100.0):
    raise NotImplementedError("implement me")
