"""Causal question: what does playing the second night of a back-to-back do to a team's result?

Unit: team-game row. Treatment: back-to-back (played yesterday). Control: 2 to 4 days of rest.
Outcomes: point margin and win. Pre-treatment covariates: own and opponent Elo, home court,
opponent's rest, how far into the season it is, season fixed effects.

Why adjustment is needed: the schedule is fixed in advance, but it is not random with respect to
who is home (second nights of a back-to-back are disproportionately road games) or who the
opponent is (the opponent's rest is correlated with ours). The naive difference mixes those in.

Checks that are NOT assumed away:
  * covariate balance before and after weighting
  * propensity overlap
  * a placebo: whether tomorrow's back-to-back "affects" today's game (it cannot, so the
    estimate should be about zero; if not, the design is picking up confounding)
  * an E-value for the win effect (how strong an unmeasured confounder must be to erase it)
Standard errors are cluster-bootstrapped by game, because both teams' rows from one game are
mirror images of each other and are not independent.

    python scripts/causal_rest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import causal as C  # noqa: E402
from nbapred import features as F  # noqa: E402
from nbapred import modern  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402

OUT = ROOT / "results" / "causal"
N_BOOT = 200
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def build_frame() -> pd.DataFrame:
    lf = F.add_team_features(prepare_long(modern.to_long(modern.games_from_team_rows(modern.load_team_rows()))))
    g = lf.groupby("game_id")
    lf["opp_rest"] = g["rest_days"].transform("sum") - lf["rest_days"]
    lf["opp_gp"] = g["season_gp"].transform("sum") - lf["season_gp"]
    lf["next_rest"] = lf.groupby("fran_id")["rest_days"].shift(-1)      # future SCHEDULE only (placebo)
    ok = ((lf.n_prior >= 10) & lf.rest_days.between(1, 4) & (lf.season_gp >= 1) & (lf.opp_gp >= 1)
          & (lf.opp_rest < 10))
    d = lf[ok].reset_index(drop=True)
    d["t"] = (d.rest_days == 1).astype(float)
    d["win"] = d["win"].astype(float)
    d["home"] = d["is_home"]
    d["opp_b2b"] = (d.opp_rest == 1).astype(float)
    d["opp_rest2"] = (d.opp_rest == 2).astype(float)
    d["opp_rest3p"] = (d.opp_rest >= 3).astype(float)
    d["own_b2b_today"] = d["t"]
    d["gp_frac"] = d["season_gp"] / 82.0
    return d


def design(d: pd.DataFrame, extra: list[str] | None = None):
    base = ["elo_i", "opp_elo_i", "home", "opp_b2b", "opp_rest2", "gp_frac"] + (extra or [])
    yrs = pd.get_dummies(d["year_id"], prefix="yr", drop_first=True).astype(float)
    cols = base + list(yrs.columns)
    X = pd.concat([d[base], yrs], axis=1)[cols].to_numpy(float)
    return X, cols


def point_and_ci(X, t, outcomes: dict[str, np.ndarray], groups, n_boot=N_BOOT):
    def est(idx):
        Xi, ti = X[idx], t[idx]
        p = C.fit_propensity(Xi, ti)(Xi)
        out = {}
        for name, y in outcomes.items():
            yi = y[idx]
            out[f"{name}|naive"] = C.naive_ate(yi, ti)
            out[f"{name}|ols"] = C.ols_ate(Xi, yi, ti)
            out[f"{name}|ipw"] = C.ipw_ate(yi, ti, p)
            out[f"{name}|aipw"] = C.aipw_ate(Xi, yi, ti, p)
        return out

    pt = est(np.arange(len(t)))
    ci = C.cluster_bootstrap(est, groups, n_boot=n_boot)
    return pt, ci


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = build_frame()
    X, cols = design(d)
    t = d["t"].to_numpy()
    groups = d["game_id"].to_numpy()
    y_m, y_w = d["margin"].to_numpy(float), d["win"].to_numpy()
    print(f"{len(d):,} team-games ({int(t.sum()):,} back-to-back, {int((1 - t).sum()):,} with 2 to 4 days rest), "
          f"{d.year_id.min()}-{d.year_id.max()}")

    # ---- balance & overlap ----
    prop = C.fit_propensity(X, t)(X)
    w = np.where(t == 1, 1 / prop, 1 / (1 - prop))
    names = cols[:6]
    smd0 = C.standardized_mean_diff(X[:, :6], t)
    smd1 = C.standardized_mean_diff(X[:, :6], t, w)
    print("\ncovariate balance (standardized mean difference, treated - control):")
    for n, a, b in zip(names, smd0, smd1):
        print(f"  {n:10s} before {a:+.3f}   after weighting {b:+.3f}")
    print(f"propensity range [{prop.min():.3f}, {prop.max():.3f}]; "
          f"{np.mean((prop < .02) | (prop > .98)):.2%} of rows trimmed")

    # ---- main estimates ----
    pt, ci = point_and_ci(X, t, {"margin": y_m, "win": y_w}, groups)
    print(f"\n{'estimate':20s} {'margin (pts)':>26s} {'win prob (pp)':>26s}")
    for m in ("naive", "ols", "ipw", "aipw"):
        a, b = pt[f"margin|{m}"], pt[f"win|{m}"]
        (alo, ahi), (blo, bhi) = ci[f"margin|{m}"], ci[f"win|{m}"]
        print(f"{m:20s} {a:+.2f} [{alo:+.2f}, {ahi:+.2f}]   {100 * b:+.1f} [{100 * blo:+.1f}, {100 * bhi:+.1f}]")

    # ---- placebo: tomorrow's back-to-back cannot affect today ----
    pl = d[d.next_rest.between(1, 4)].reset_index(drop=True)
    Xp, _ = design(pl, extra=["own_b2b_today"])
    tp = (pl.next_rest == 1).astype(float).to_numpy()
    ptp, cip = point_and_ci(Xp, tp, {"margin": pl["margin"].to_numpy(float), "win": pl["win"].to_numpy()},
                            pl["game_id"].to_numpy())
    print(f"\nPLACEBO (next game is a back-to-back; n={len(pl):,}), AIPW:")
    print(f"  margin {ptp['margin|aipw']:+.2f} [{cip['margin|aipw'][0]:+.2f}, {cip['margin|aipw'][1]:+.2f}]   "
          f"win {100 * ptp['win|aipw']:+.1f}pp [{100 * cip['win|aipw'][0]:+.1f}, {100 * cip['win|aipw'][1]:+.1f}]")

    # ---- heterogeneity (AIPW, margin) ----
    subs = {
        "home games": d.home == 1, "road games": d.home == 0,
        "opponent also on B2B": d.opp_b2b == 1, "opponent rested": d.opp_b2b == 0,
        "2011-2017": d.year_id <= 2017, "2018-2024": d.year_id >= 2018,
    }
    het = {}
    print("\nHETEROGENEITY (AIPW margin):")
    for name, mask in subs.items():
        m = mask.to_numpy()
        Xs, _ = design(d[m].reset_index(drop=True))
        ps, cs = point_and_ci(Xs, t[m], {"margin": y_m[m], "win": y_w[m]}, groups[m], n_boot=100)
        het[name] = {"n": int(m.sum()), "margin": ps["margin|aipw"], "margin_ci": cs["margin|aipw"],
                     "win": ps["win|aipw"], "win_ci": cs["win|aipw"]}
        lo, hi = cs["margin|aipw"]
        print(f"  {name:22s} n={m.sum():6,}  {ps['margin|aipw']:+.2f} [{lo:+.2f}, {hi:+.2f}] pts   "
              f"{100 * ps['win|aipw']:+.1f}pp")

    # ---- E-value on the win effect ----
    p0 = float(y_w[t == 0].mean())
    ate = pt["win|aipw"]
    rr = (p0 + ate) / p0
    lo_w, hi_w = ci["win|aipw"]
    near_null = hi_w if ate < 0 else lo_w
    rr_ci = (p0 + near_null) / p0
    ev = C.e_value(rr)
    ev_ci = 1.0 if (ate < 0 < near_null) or (ate > 0 > near_null) else C.e_value(rr_ci)
    print(f"\nE-value: control win rate {p0:.3f}, RR {rr:.3f} -> E-value {ev:.2f} "
          f"(for the CI bound nearest zero: {ev_ci:.2f}; 1.00 means the CI includes no effect)")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y0 = np.arange(len(names))
    ax.scatter(np.abs(smd0), y0, label="before", color="#c0392b")
    ax.scatter(np.abs(smd1), y0, label="after IPW", color="#1f6f8b")
    ax.axvline(0.1, ls=":", color="#999")
    ax.set_yticks(y0, names); ax.set_xlabel("|standardized mean difference|"); ax.legend()
    ax.set_title("Covariate balance (back-to-back vs rested)")
    fig.tight_layout(); fig.savefig(OUT / "fig_causal_balance.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(prop[t == 0], bins=40, alpha=0.6, label="rested (control)", density=True)
    ax.hist(prop[t == 1], bins=40, alpha=0.6, label="back-to-back (treated)", density=True)
    ax.set_xlabel("propensity score P(back-to-back | covariates)"); ax.legend(); ax.set_title("Overlap")
    fig.tight_layout(); fig.savefig(OUT / "fig_causal_overlap.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    rows = [(f"{m}", pt[f"margin|{m}"], *ci[f"margin|{m}"]) for m in ("naive", "ols", "ipw", "aipw")]
    rows.append(("PLACEBO aipw", ptp["margin|aipw"], *cip["margin|aipw"]))
    for i, (lab, v, lo, hi) in enumerate(rows):
        col = "#999" if "PLACEBO" in lab else "#1f6f8b" if lab == "aipw" else "#9aa5b1"
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=col, capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis(); ax.axvline(0, color="#555", lw=1)
    ax.set_xlabel("effect of back-to-back on point margin (95% cluster-bootstrap CI)")
    ax.set_title("Estimates by method, and the placebo")
    fig.tight_layout(); fig.savefig(OUT / "fig_causal_forest.png", dpi=160); plt.close(fig)

    (OUT / "causal_results.json").write_text(json.dumps({
        "n": len(d), "n_treated": int(t.sum()), "estimates": {k: {"point": v, "ci": ci[k]} for k, v in pt.items()},
        "placebo": {k: {"point": v, "ci": cip[k]} for k, v in ptp.items()}, "heterogeneity": het,
        "control_win_rate": p0, "e_value": ev, "e_value_ci": ev_ci,
        "balance": {n: {"before": float(a), "after": float(b)} for n, a, b in zip(names, smd0, smd1)},
    }, indent=2, default=float))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
