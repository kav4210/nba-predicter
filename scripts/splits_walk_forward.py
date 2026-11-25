"""Do home/road splits or 'what a team gives up' features add anything over Elo? Walk-forward, 2017-2024.

Hypotheses, written before the run:
  H4  Neither opponent-allowed rates (st_allow_*) nor same-venue margin (st_venue_margin) improves
      logreg, hist_gb or MLP over the Elo feature set: every paired CI straddles zero and |delta| < 0.001.
      Reason: Elo and last-10 margin already summarise team quality, and home advantage is a league-wide
      constant that the home flag and league home rate already carry.
  H5  Gradient boosting is hurt (not helped) by the extra columns, as it was by style features.
Same split recipe as walk_forward.py (train <= s-2, validate s-1, test s).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nbapred import features as F  # noqa: E402
from nbapred import metrics as M  # noqa: E402
from nbapred import modern, style  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402
from nbapred.models import MLP  # noqa: E402
from nbapred.train import fit_temperature, fit_torch, predict_logits  # noqa: E402

OUT = ROOT / "results" / "splits"
TEST_SEASONS = list(range(2017, 2025))
torch.set_num_threads(2)
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731


def trio(base: list[str], names: list[str]) -> list[str]:
    return [f"{p}_st_{n}" for n in names for p in ("h", "a", "d")] if names else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    completed = modern.games_from_team_rows(modern.load_team_rows())
    lf = style.add_trailing_style(F.add_team_features(prepare_long(modern.to_long(completed))))
    games = F.build_games(lf)
    games = games[F.eligible(games)].reset_index(drop=True)
    y = games["y"].to_numpy().astype(np.float32)
    base = F.feature_columns(True)
    sets = {"elo": base,
            "elo+allow": base + trio(base, style.ALLOW_RAW),
            "elo+venue": base + trio(base, ["venue_margin"]),
            "elo+both": base + trio(base, style.SPLIT_RAW)}
    probs: dict[str, list] = {}
    ys = []
    for s in TEST_SEASONS:
        tr = np.flatnonzero((games.year_id <= s - 2).to_numpy())
        va = np.flatnonzero((games.year_id == s - 1).to_numpy())
        te = np.flatnonzero((games.year_id == s).to_numpy())
        ys.append(y[te])
        for fs, cols in sets.items():
            X = games[cols].to_numpy(float)
            imp = SimpleImputer(strategy="median").fit(X[tr])
            scl = StandardScaler().fit(imp.transform(X[tr]))
            Xs = scl.transform(imp.transform(X)).astype(np.float32)
            best = None
            for C in (0.01, 0.1, 1.0):
                lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
                ll = M.log_loss(lr.predict_proba(Xs[va])[:, 1], y[va])
                if best is None or ll < best[0]:
                    best = (ll, lr)
            probs.setdefault(f"logreg[{fs}]", []).append(best[1].predict_proba(Xs[te])[:, 1])
            kw = dict(learning_rate=0.04, max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
                      early_stopping=False, random_state=0)
            probe = HistGradientBoostingClassifier(max_iter=400, **kw).fit(X[tr], y[tr])
            n = int(np.argmin([M.log_loss(p[:, 1], y[va]) for p in probe.staged_predict_proba(X[va])])) + 1
            probs.setdefault(f"hist_gb[{fs}]", []).append(
                HistGradientBoostingClassifier(max_iter=n, **kw).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            lv, lt = [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                net, _ = fit_torch(MLP(Xs.shape[1]), (tt(Xs[tr]), tt(y[tr])), (tt(Xs[va]), tt(y[va])), seed=seed)
                lv.append(predict_logits(net, (tt(Xs[va]),))); lt.append(predict_logits(net, (tt(Xs[te]),)))
            T = fit_temperature(np.mean(lv, 0), y[va])
            probs.setdefault(f"mlp[{fs}]", []).append(sig(np.mean(lt, 0) / T))
        print(f"fold {s} done [{time.time() - t0:.0f}s]", flush=True)

    P = {k: np.concatenate(v) for k, v in probs.items()}
    Y = np.concatenate(ys)
    res = {k: M.summarize(v, Y) for k, v in P.items()}
    print(f"\n=== {len(Y):,} games, test seasons {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} ===")
    for k, r in sorted(res.items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{k:24s} {r['log_loss']:.4f} acc {r['accuracy']:.3f} ece {r['ece']:.3f}")
    cmp = {}
    print("\nPAIRED (negative = extra features help):")
    for model in ("logreg", "hist_gb", "mlp"):
        for fs in ("elo+allow", "elo+venue", "elo+both"):
            d, lo, hi = M.paired_bootstrap_delta(P[f"{model}[{fs}]"], P[f"{model}[elo]"], Y)
            cmp[f"{model} {fs} vs elo"] = {"delta": d, "ci": [lo, hi]}
            print(f"  {model:8s} {fs:10s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (OUT / "splits_results.json").write_text(json.dumps({"n_games": int(len(Y)), "results": res, "comparisons": cmp}, indent=1))


if __name__ == "__main__":
    main()
