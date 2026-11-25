"""Team playing-style metrics from box scores, and an unsupervised (PCA) compression of them.

Per-game metrics describe how a team played; the trailing versions (st_*) average the previous
20 games with shift(1), so a game's own box score never enters its own features. PCA is fit on
training rows only. tests/test_style.py rewrites the future (own game included) and checks that
nothing moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# efficiency (ortg, drtg, efg) + style (the rest)
STYLE_RAW = ["pace", "three_rate", "ft_rate", "tov_rate", "oreb_pct", "dreb_pct", "ast_rate", "efg", "ortg", "drtg"]
# metrics that describe HOW a team plays, not how good it is; used for archetype clustering
ARCHETYPE = ["pace", "three_rate", "ft_rate", "tov_rate", "oreb_pct", "dreb_pct", "ast_rate", "stl_rate", "blk_rate"]
# what a team gives up (the opponent's rates against it) and how it does at its current venue
ALLOW_RAW = ["allow_efg", "allow_three_rate", "allow_ft_rate", "allow_oreb_pct", "allow_ast_rate", "allow_tov_rate"]
SPLIT_RAW = ALLOW_RAW + ["venue_margin"]
ALL_METRICS = sorted(set(STYLE_RAW) | set(ARCHETYPE), key=(STYLE_RAW + ARCHETYPE).index)


def game_metrics(long: pd.DataFrame) -> pd.DataFrame:
    """Per team-game metrics (about that game only; callers must lag them before using as features)."""
    b = lambda c: long[f"box_{c}"]   # noqa: E731  own box score
    o = lambda c: long[f"obox_{c}"]  # noqa: E731  opponent box score
    poss = b("FGA") - b("OREB") + b("TOV") + 0.44 * b("FTA")
    opp_poss = o("FGA") - o("OREB") + o("TOV") + 0.44 * o("FTA")
    m = pd.DataFrame(index=long.index)
    m["pace"] = (poss + opp_poss) / 2.0
    m["three_rate"] = b("FG3A") / b("FGA")
    m["ft_rate"] = b("FTA") / b("FGA")
    m["tov_rate"] = b("TOV") / poss
    m["oreb_pct"] = b("OREB") / (b("OREB") + o("DREB"))
    m["dreb_pct"] = b("DREB") / (b("DREB") + o("OREB"))
    m["ast_rate"] = b("AST") / b("FGM")
    m["efg"] = (b("FGM") + 0.5 * b("FG3M")) / b("FGA")
    m["stl_rate"] = b("STL") / opp_poss
    m["blk_rate"] = b("BLK") / o("FGA")
    m["ortg"] = 100.0 * long["pts"] / poss
    m["drtg"] = 100.0 * long["opp_pts"] / opp_poss
    m["allow_efg"] = (o("FGM") + 0.5 * o("FG3M")) / o("FGA")
    m["allow_three_rate"] = o("FG3A") / o("FGA")
    m["allow_ft_rate"] = o("FTA") / o("FGA")
    m["allow_oreb_pct"] = o("OREB") / (o("OREB") + b("DREB"))
    m["allow_ast_rate"] = o("AST") / o("FGM")
    m["allow_tov_rate"] = o("TOV") / opp_poss
    return m


def add_trailing_style(long_feat: pd.DataFrame, window: int = 20, min_periods: int = 10) -> pd.DataFrame:
    """Adds st_<metric>: mean of the team's previous `window` games. No-op without box scores."""
    if "box_FGA" not in long_feat.columns:
        return long_feat
    out = long_feat.copy()
    m = game_metrics(out)
    grp = out["fran_id"]
    for c in STYLE_RAW + ALLOW_RAW:
        out[f"st_{c}"] = m[c].groupby(grp, sort=False).transform(
            lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    # same-venue form: mean margin over the team's previous 10 games at this venue type (home or road)
    out["st_venue_margin"] = out["margin"].groupby([grp, out["is_home"]], sort=False).transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    return out


def style_cols(side: str) -> list[str]:
    return [f"{side}_st_{c}" for c in STYLE_RAW]


class StylePCA:
    """Fit on training games only: stack home and away trailing-style vectors, standardise, PCA."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, games: pd.DataFrame) -> "StylePCA":
        X = np.vstack([games[style_cols("h")].to_numpy(float), games[style_cols("a")].to_numpy(float)])
        self.imp = SimpleImputer(strategy="median").fit(X)
        self.sc = StandardScaler().fit(self.imp.transform(X))
        self.pca = PCA(n_components=self.k, random_state=0).fit(self.sc.transform(self.imp.transform(X)))
        return self

    def _proj(self, games: pd.DataFrame, side: str) -> np.ndarray:
        X = games[style_cols(side)].to_numpy(float)
        return self.pca.transform(self.sc.transform(self.imp.transform(X)))

    def transform(self, games: pd.DataFrame) -> pd.DataFrame:
        h, a = self._proj(games, "h"), self._proj(games, "a")
        cols = {}
        for i in range(self.k):
            cols[f"h_pc{i + 1}"], cols[f"a_pc{i + 1}"], cols[f"d_pc{i + 1}"] = h[:, i], a[:, i], h[:, i] - a[:, i]
        return pd.DataFrame(cols, index=games.index)

    def pc_names(self) -> list[str]:
        return [f"{p}_pc{i + 1}" for i in range(self.k) for p in ("h", "a", "d")]
