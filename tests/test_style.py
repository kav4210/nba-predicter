"""Style features must not see the game's own box score or anything after it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbapred import features as F  # noqa: E402
from nbapred import modern, style  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402


@pytest.fixture(scope="module")
def completed():
    return modern.games_from_team_rows(modern.load_team_rows())


def build(games):
    lf = style.add_trailing_style(F.add_team_features(prepare_long(modern.to_long(games))))
    return F.build_games(lf)


def test_style_features_ignore_own_and_future_box_scores(completed):
    cutoff = pd.Timestamp("2019-01-15")
    assert (completed["date"] == cutoff).sum() >= 4
    t = completed.copy()
    m = (t["date"] >= cutoff).to_numpy()
    rng = np.random.default_rng(0)
    for c in [c for c in t.columns if c[:2] in ("h_", "a_")] + ["home_pts", "away_pts"]:
        t.loc[m, c] = t.loc[m, c].to_numpy() * rng.uniform(0.5, 1.5, m.sum())

    base, new = build(completed), build(t)
    cols = [c for c in base.columns if "_st_" in c] + F.feature_columns(True)
    a = base[base.date <= cutoff].set_index("game_id")[cols]
    b = new[new.date <= cutoff].set_index("game_id").loc[a.index, cols]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)

    # teeth: tampered games after the cutoff really do get different style features
    later_a = base[base.date > cutoff + pd.Timedelta(days=60)].set_index("game_id")["h_st_pace"]
    later_b = new[new.date > cutoff + pd.Timedelta(days=60)].set_index("game_id").loc[later_a.index, "h_st_pace"]
    assert not np.allclose(later_a.fillna(0), later_b.fillna(0))


def test_style_pca_is_fit_on_given_rows_only(completed):
    g = build(completed)
    g = g[F.eligible(g)]
    train = g[g.year_id <= 2018]
    p = style.StylePCA(k=3).fit(train)
    out = p.transform(g)
    assert out.shape == (len(g), 9)
    assert abs(p.pca.mean_).max() < 1e-6  # centred on training rows (post-standardisation)
    assert p.pca.explained_variance_ratio_.sum() > 0.4
