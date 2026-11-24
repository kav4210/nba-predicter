"""Train and compare models for NBA home-win probability.

Chronological split (no shuffling across time):
    train: seasons <= 2009 | validation: 2010-2011 | test: 2012-2015
All tuning (epochs, trees, C, temperature) uses validation only. The test seasons are
not evaluated unless --final is passed, so there is one honest look at them.

    python scripts/run_experiment.py --seeds 1           # dev: validation only
    python scripts/run_experiment.py --seeds 5 --final   # final: also test + figures
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import features as F  # noqa: E402
from nbapred import metrics as M  # noqa: E402
from nbapred.data import load_raw, prepare_long  # noqa: E402
from nbapred.models import MLP, SeqNet  # noqa: E402
from nbapred.train import fit_temperature, fit_torch, predict_logits  # noqa: E402

SPLITS = {"538": (2009, 2011), "modern": (2021, 2022)}  # (train end, val end); season = ending year
torch.set_num_threads(2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tt(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--final", action="store_true", help="also evaluate on the test seasons")
    ap.add_argument("--source", choices=["538", "modern"], default="538")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    TRAIN_END, VAL_END = SPLITS[args.source]
    out = Path(args.out) if args.out else ROOT / "results" / ("modern" if args.source == "modern" else "")
    out.mkdir(parents=True, exist_ok=True)
    BENCH = "538_elo_forecast" if args.source == "538" else "own_elo_forecast"

    t0 = time.time()
    if args.source == "538":
        long_feat = F.add_team_features(prepare_long(load_raw()))
    else:
        from nbapred import modern
        completed = modern.games_from_team_rows(modern.load_team_rows())
        long_feat = F.add_team_features(prepare_long(modern.to_long(completed)))
    games = F.build_games(long_feat)
    games = games[F.eligible(games)].reset_index(drop=True)
    split = np.where(games.year_id <= TRAIN_END, "train", np.where(games.year_id <= VAL_END, "val", "test"))
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    y = games["y"].to_numpy().astype(np.float32)
    print({s: len(i) for s, i in idx.items()}, f"home win rate (train) = {y[idx['train']].mean():.3f}")

    splits_eval = ["val", "test"] if args.final else ["val"]
    probs: dict[str, dict[str, np.ndarray]] = {s: {} for s in splits_eval}
    info: dict = {}

    # ---- benchmarks ------------------------------------------------------------------
    base_rate = y[idx["train"]].mean()
    for s in splits_eval:
        probs[s]["always_home_rate"] = np.full(len(idx[s]), base_rate)
        probs[s][BENCH] = games["elo_forecast"].to_numpy()[idx[s]]

    # Control: is the gain over 538 just re-fitting home-court advantage? Elo gap + league home rate only.
    ctl_cols = ["d_elo", "lg_home_rate", "is_neutral"]
    Xc = games[ctl_cols].to_numpy(dtype=np.float64)
    sc_c = StandardScaler().fit(Xc[idx["train"]])
    ctl = LogisticRegression(max_iter=1000).fit(sc_c.transform(Xc[idx["train"]]), y[idx["train"]])
    for s in splits_eval:
        probs[s]["CONTROL logreg[elo gap + home rate only]"] = ctl.predict_proba(sc_c.transform(Xc[idx[s]]))[:, 1]

    for use_elo in (False, True):
        tag = "elo" if use_elo else "noelo"
        cols = F.feature_columns(use_elo)
        X = games[cols].to_numpy(dtype=np.float64)
        imp = SimpleImputer(strategy="median").fit(X[idx["train"]])
        sc = StandardScaler().fit(imp.transform(X[idx["train"]]))
        Xs = sc.transform(imp.transform(X)).astype(np.float32)

        # ---- logistic regression (C picked on validation) ------------------------------
        best = None
        for C in (0.01, 0.1, 1.0):
            lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[idx["train"]], y[idx["train"]])
            ll = M.log_loss(lr.predict_proba(Xs[idx["val"]])[:, 1], y[idx["val"]])
            if best is None or ll < best[0]:
                best = (ll, C, lr)
        for s in splits_eval:
            probs[s][f"logreg[{tag}]"] = best[2].predict_proba(Xs[idx[s]])[:, 1]
        info[f"logreg[{tag}]"] = {"C": best[1]}

        # ---- gradient boosting: pick #trees on validation, refit once -------------------
        hgb_kw = dict(learning_rate=0.04, max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
                      early_stopping=False, random_state=0)
        probe = HistGradientBoostingClassifier(max_iter=600, **hgb_kw).fit(X[idx["train"]], y[idx["train"]])
        val_ll = [M.log_loss(p[:, 1], y[idx["val"]]) for p in probe.staged_predict_proba(X[idx["val"]])]
        n_trees = int(np.argmin(val_ll)) + 1
        hgb = HistGradientBoostingClassifier(max_iter=n_trees, **hgb_kw).fit(X[idx["train"]], y[idx["train"]])
        for s in splits_eval:
            probs[s][f"hist_gb[{tag}]"] = hgb.predict_proba(X[idx[s]])[:, 1]
        info[f"hist_gb[{tag}]"] = {"n_trees": n_trees}
        if use_elo:
            pi = permutation_importance(hgb, X[idx["val"]], y[idx["val"]], scoring="neg_log_loss",
                                        n_repeats=5, random_state=0)
            order = np.argsort(-pi.importances_mean)[:12]
            info["hgb_perm_importance_val"] = [(cols[i], float(pi.importances_mean[i])) for i in order]

        # ---- PyTorch: MLP and GRU sequence net, multi-seed -----------------------------
        seq_h = F.build_sequences(long_feat, games["home_rowid"].to_numpy(), use_elo)
        seq_a = F.build_sequences(long_feat, games["away_rowid"].to_numpy(), use_elo)
        step_dim = seq_h.shape[2]
        # standardise sequence channels with train statistics
        flat = seq_h[idx["train"]].reshape(-1, step_dim)
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        seq_h, seq_a = (seq_h - mu) / sd, (seq_a - mu) / sd

        def pack(kind, s):
            i = idx[s]
            if kind == "mlp":
                return (tt(Xs[i]), tt(y[i]))
            return (tt(seq_h[i]), tt(seq_a[i]), tt(Xs[i]), tt(y[i]))

        for kind in ("mlp", "seqnet"):
            name = f"{kind}[{tag}]"
            logits = {s: [] for s in splits_eval}
            hists = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = MLP(Xs.shape[1]) if kind == "mlp" else SeqNet(step_dim + 0, Xs.shape[1])
                model, hist = fit_torch(model, pack(kind, "train"), pack(kind, "val"), seed=seed)
                hists.append(hist)
                for s in splits_eval:
                    logits[s].append(predict_logits(model, pack(kind, s)[:-1]))
            ens = {s: np.mean(logits[s], axis=0) for s in splits_eval}  # average logits over seeds
            T = fit_temperature(ens["val"], y[idx["val"]])  # calibration fitted on validation only
            for s in splits_eval:
                probs[s][name] = sigmoid(ens[s] / T)
            info[name] = {"temperature": T, "best_epochs": [h["best_epoch"] for h in hists],
                          "history_seed0": {"train": hists[0]["train"], "val": hists[0]["val"]}}
            print(f"  {name}: T={T:.3f} epochs={info[name]['best_epochs']}  [{time.time() - t0:.0f}s]")

    # ---- pre-declared headline model: average of the three learned models with Elo -----
    for s in splits_eval:
        probs[s]["ENSEMBLE[hist_gb+mlp+seqnet, elo]"] = np.mean(
            [probs[s]["hist_gb[elo]"], probs[s]["mlp[elo]"], probs[s]["seqnet[elo]"]], axis=0)

    # ---- report ------------------------------------------------------------------------
    results = {}
    for s in splits_eval:
        ys = y[idx[s]]
        results[s] = {k: M.summarize(v, ys) for k, v in probs[s].items()}
        ref = probs[s][BENCH]
        for k, v in probs[s].items():
            d, lo, hi = M.paired_bootstrap_delta(v, ref, ys)
            results[s][k].update({"dlogloss_vs_bench": d, "ci_lo": lo, "ci_hi": hi})
        print(f"\n=== {s.upper()} ({len(ys)} games) ===")
        print(f"{'model':38s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'ece':>6s}  dLL vs bench [95% CI]")
        for k, r in sorted(results[s].items(), key=lambda kv: kv[1]["log_loss"]):
            print(f"{k:38s} {r['log_loss']:8.4f} {r['brier']:7.4f} {r['accuracy']:6.3f} {r['ece']:6.3f}"
                  f"  {r['dlogloss_vs_bench']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

    if args.final:
        head = "ENSEMBLE[hist_gb+mlp+seqnet, elo]"
        seasons = games["year_id"].to_numpy()[idx["test"]]
        per_season = {}
        for sea in sorted(set(seasons)):
            m = seasons == sea
            per_season[int(sea)] = {
                "n": int(m.sum()),
                "headline_logloss": M.log_loss(probs["test"][head][m], y[idx["test"]][m]),
                "bench_logloss": M.log_loss(probs["test"][BENCH][m], y[idx["test"]][m]),
                "headline_acc": M.accuracy(probs["test"][head][m], y[idx["test"]][m]),
                "bench_acc": M.accuracy(probs["test"][BENCH][m], y[idx["test"]][m]),
            }
        info["per_season_test"] = per_season
        (out / "results.json").write_text(json.dumps({"results": results, "info": info}, indent=2))
        np.savez(out / "test_probs.npz", y=y[idx["test"]], **{k: v for k, v in probs["test"].items()})
        from make_figures import make_figures  # noqa: E402
        make_figures(results, info, probs, y[idx["test"]], out, bench=BENCH)
        print(f"\nwrote results to {out}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
