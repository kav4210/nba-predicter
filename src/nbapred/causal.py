"""Observational causal-effect estimators for a binary treatment.

Setup: treatment t in {0,1}, outcome y, pre-treatment covariates X. Target = average treatment
effect E[Y(1) - Y(0)]. Identification needs (1) no unmeasured confounding given X,
(2) overlap: 0 < P(t=1|X) < 1, (3) consistency. None of these is testable from data alone;
`e_value` quantifies how strong an unmeasured confounder would have to be to erase an estimate.

Estimators: naive difference in means, OLS adjustment, IPW (Hajek / normalised), and AIPW
(doubly robust: consistent if either the propensity model or the outcome models are right).
Uncertainty: cluster bootstrap (resample whole groups, e.g. games, so correlated rows stay together).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


def standardized_mean_diff(X: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """(mean_treated - mean_control) / pooled sd, per column; optionally with weights."""
    w = np.ones(len(t)) if w is None else w
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        s = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2.0)  # unweighted pooled sd, the usual convention
        out.append((m1 - m0) / s if s > 0 else 0.0)
    return np.array(out)


def fit_propensity(X: np.ndarray, t: np.ndarray, C: float = 1.0):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X), t)
    return lambda Z: lr.predict_proba(sc.transform(Z))[:, 1]


def naive_ate(y, t) -> float:
    return float(y[t == 1].mean() - y[t == 0].mean())


def ols_ate(X, y, t) -> float:
    """Coefficient on t in y ~ t + X."""
    return float(LinearRegression().fit(np.column_stack([t, X]), y).coef_[0])


def ipw_ate(y, t, p, trim=(0.02, 0.98)) -> float:
    """Normalised (Hajek) inverse-probability-weighted ATE, with propensity trimming."""
    keep = (p >= trim[0]) & (p <= trim[1])
    y, t, p = y[keep], t[keep], p[keep]
    w1, w0 = t / p, (1 - t) / (1 - p)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def aipw_ate(X, y, t, p, trim=(0.02, 0.98)) -> float:
    """Augmented IPW: outcome-model prediction plus an inverse-weighted residual correction."""
    keep = (p >= trim[0]) & (p <= trim[1])
    X, y, t, p = X[keep], y[keep], t[keep], p[keep]
    m1 = LinearRegression().fit(X[t == 1], y[t == 1])
    m0 = LinearRegression().fit(X[t == 0], y[t == 0])
    mu1, mu0 = m1.predict(X), m0.predict(X)
    return float(np.mean(mu1 - mu0 + t * (y - mu1) / p - (1 - t) * (y - mu0) / (1 - p)))


def estimate_all(X, y, t) -> dict[str, float]:
    p = fit_propensity(X, t)(X)
    return {"naive": naive_ate(y, t), "ols": ols_ate(X, y, t), "ipw": ipw_ate(y, t, p), "aipw": aipw_ate(X, y, t, p)}


def cluster_bootstrap(fn, groups: np.ndarray, n_boot: int = 200, seed: int = 0, alpha: float = 0.05):
    """`fn(row_index) -> dict[name, float]`. Resamples whole groups with replacement.

    Returns {name: (lo, hi)} percentile intervals.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        for k, v in fn(idx).items():
            draws.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
            for k, v in draws.items()}


def e_value(rr: float) -> float:
    """VanderWeele-Ding E-value: minimum risk-ratio strength an unmeasured confounder needs with BOTH
    treatment and outcome to fully explain away an observed risk ratio. rr<1 is inverted first."""
    rr = 1.0 / rr if rr < 1 else rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))
