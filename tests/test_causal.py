"""Estimators must recover a KNOWN effect from synthetic confounded data, and the naive one must not."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbapred import causal as C  # noqa: E402

TRUE = -1.5


def simulate(n=20000, seed=0, confound=1.0, nonlinear=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logit = confound * ((1.2 if nonlinear else 0.9) * X[:, 0] - 0.5 * X[:, 1])
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    base = 2.0 * X[:, 0] + 1.0 * X[:, 1] + (4.0 * np.sin(2.0 * X[:, 0]) if nonlinear else 0)
    y = base + TRUE * t + rng.normal(size=n)
    return X, y, t


def test_naive_is_biased_adjusted_estimators_are_not():
    X, y, t = simulate()
    est = C.estimate_all(X, y, t)
    assert abs(est["naive"] - TRUE) > 1.0            # confounding makes the naive estimate badly wrong
    for k in ("ols", "ipw", "aipw"):
        assert abs(est[k] - TRUE) < 0.25, (k, est)


def test_doubly_robust_survives_a_wrong_outcome_model_ols_does_not():
    # the outcome is sine-shaped in X0, so every linear outcome model is misspecified;
    # the propensity model (logistic in X) is correct, which is all IPW and AIPW need
    X, y, t = simulate(n=30000, nonlinear=True, seed=1)
    est = C.estimate_all(X, y, t)
    assert abs(est["ols"] - TRUE) > 0.3           # adjustment with a wrong outcome model stays biased
    assert abs(est["ipw"] - TRUE) < 0.2
    assert abs(est["aipw"] - TRUE) < 0.2          # the residual-correction term is what rescues it


def test_weighting_balances_covariates():
    X, y, t = simulate()
    p = C.fit_propensity(X, t)(X)
    before = np.abs(C.standardized_mean_diff(X, t))
    w = np.where(t == 1, 1 / p, 1 / (1 - p))
    after = np.abs(C.standardized_mean_diff(X, t, w))
    assert before.max() > 0.3 and after.max() < 0.1


def test_cluster_bootstrap_interval_covers_truth_and_is_wider_than_naive_iid():
    X, y, t = simulate(n=6000, seed=2)
    groups = np.repeat(np.arange(3000), 2)  # rows come in correlated pairs, like two sides of one game
    ci = C.cluster_bootstrap(lambda i: {"aipw": C.estimate_all(X[i], y[i], t[i])["aipw"]}, groups, n_boot=60)
    lo, hi = ci["aipw"]
    assert lo < TRUE < hi


def test_no_effect_is_recovered_as_zero_and_placebo_logic_holds():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20000, 2))
    t = (rng.uniform(size=20000) < 1 / (1 + np.exp(-X[:, 0]))).astype(float)
    y = 2 * X[:, 0] + rng.normal(size=20000)  # treatment does nothing
    est = C.estimate_all(X, y, t)
    assert abs(est["aipw"]) < 0.1 and abs(est["naive"]) > 0.5


def test_e_value_known_values():
    assert C.e_value(1.0) == pytest.approx(1.0)
    assert C.e_value(2.0) == pytest.approx(2 + np.sqrt(2))   # 3.414...
    assert C.e_value(0.5) == pytest.approx(C.e_value(2.0))   # protective effects are inverted
