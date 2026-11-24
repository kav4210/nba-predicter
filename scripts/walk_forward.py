"""Walk-forward evaluation: retrain every season and test on the next one.

Fold for test season s: train = seasons <= s-2, validation = s-1 (early stopping, trees, C,
temperature), test = s. Test seasons 2017-2024 give about 9,500 held-out games, roughly 4x the
single 2023-24 split, so confidence intervals are about half as wide.

HYPOTHESES, written before running (outcomes are reported against these, not reinterpreted):
  H1  Style PCA features add at most 0.002 log loss over Elo+form; the CI includes zero.
  H2  The transformer is within +/-0.003 log loss of both the GRU and plain logistic regression.
  H3  Averaging logreg + gradient boosting + MLP beats the best single model by at most 0.002,
      and beats the two-feature control by 0.002 to 0.004.

    python scripts/walk_forward.py --seeds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import features as F  # noqa: E402
from nbapred import metrics as M  # noqa: E402
from nbapred import modern, style  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402
from nbapred.models import MLP, SeqNet, TransNet  # noqa: E402
from nbapred.train import fit_temperature, fit_torch, predict_logits  # noqa: E402

OUT = ROOT / "results" / "walk_forward"
TEST_SEASONS = list(range(2017, 2025))
torch.set_num_threads(2)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
tt = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731

BENCH = "own_elo_forecast"
CONTROL = "CONTROL logreg[elo gap + home rate]"
ENS = "ENSEMBLE3[logreg+hist_gb+mlp, elo]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=len(TEST_SEASONS), help="use only the last N test seasons (debug)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    completed = modern.games_from_team_rows(modern.load_team_rows())
    lf = style.add_trailing_style(F.add_team_features(prepare_long(modern.to_long(completed))))
    games = F.build_games(lf)
    games = games[F.eligible(games)].reset_index(drop=True)
    y = games["y"].to_numpy().astype(np.float32)
    seq_h = F.build_sequences(lf, games["home_rowid"].to_numpy(), True)
    seq_a = F.build_sequences(lf, games["away_rowid"].to_numpy(), True)
    step_dim = seq_h.shape[2]
    base_cols = F.feature_columns(True)
    ctl_cols = ["d_elo", "lg_home_rate", "is_neutral"]
    print(f"{len(games):,} eligible games, seasons {games.year_id.min()}-{games.year_id.max()}", flush=True)

    probs: dict[str, list[np.ndarray]] = {}
    ys, seasons = [], []
    info: dict = {"folds": {}}
    last_trans = None

    def add(name, p):
        probs.setdefault(name, []).append(np.asarray(p))

    for s in TEST_SEASONS[-args.folds:]:
        tf = time.time()
        tr = np.flatnonzero((games.year_id <= s - 2).to_numpy())
        va = np.flatnonzero((games.year_id == s - 1).to_numpy())
        te = np.flatnonzero((games.year_id == s).to_numpy())
        ys.append(y[te]); seasons.append(np.full(len(te), s))
        add(BENCH, games["elo_forecast"].to_numpy()[te])

        pca = style.StylePCA(k=5).fit(games.iloc[tr])
        pcs = pca.transform(games)
        feats = {"elo": games[base_cols], "elo+style": pd.concat([games[base_cols], pcs], axis=1)}

        # control: Elo gap + league home rate
        sc = StandardScaler().fit(games[ctl_cols].iloc[tr])
        ctl = LogisticRegression(max_iter=1000).fit(sc.transform(games[ctl_cols].iloc[tr]), y[tr])
        add(CONTROL, ctl.predict_proba(sc.transform(games[ctl_cols].iloc[te]))[:, 1])

        ens_parts = {}
        for fs, df in feats.items():
            X = df.to_numpy(float)
            imp = SimpleImputer(strategy="median").fit(X[tr])
            scl = StandardScaler().fit(imp.transform(X[tr]))
            Xs = scl.transform(imp.transform(X)).astype(np.float32)

            best = None
            for C in (0.01, 0.1, 1.0):
                lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
                ll = M.log_loss(lr.predict_proba(Xs[va])[:, 1], y[va])
                if best is None or ll < best[0]:
                    best = (ll, lr)
            p_lr = best[1].predict_proba(Xs[te])[:, 1]
            add(f"logreg[{fs}]", p_lr)

            kw = dict(learning_rate=0.04, max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
                      early_stopping=False, random_state=0)
            probe = HistGradientBoostingClassifier(max_iter=400, **kw).fit(X[tr], y[tr])
            n_trees = int(np.argmin([M.log_loss(p[:, 1], y[va]) for p in probe.staged_predict_proba(X[va])])) + 1
            p_gb = HistGradientBoostingClassifier(max_iter=n_trees, **kw).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            add(f"hist_gb[{fs}]", p_gb)

            lv, lt = [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                net, _ = fit_torch(MLP(Xs.shape[1]), (tt(Xs[tr]), tt(y[tr])), (tt(Xs[va]), tt(y[va])), seed=seed)
                lv.append(predict_logits(net, (tt(Xs[va]),))); lt.append(predict_logits(net, (tt(Xs[te]),)))
            T = fit_temperature(np.mean(lv, 0), y[va])
            p_mlp = sig(np.mean(lt, 0) / T)
            add(f"mlp[{fs}]", p_mlp)
            if fs == "elo":
                ens_parts = (p_lr, p_gb, p_mlp)
                Xs_elo = Xs

        add(ENS, np.mean(ens_parts, axis=0))

        # sequence models (Elo feature set): GRU vs transformer, same fusion head and training recipe
        flat = seq_h[tr].reshape(-1, step_dim)
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        sh, sa = (seq_h - mu) / sd, (seq_a - mu) / sd
        pk = lambda idx: (tt(sh[idx]), tt(sa[idx]), tt(Xs_elo[idx]))  # noqa: E731
        for name, make in (("gru[elo]", lambda: SeqNet(step_dim, Xs_elo.shape[1])),
                           ("transformer[elo]", lambda: TransNet(step_dim, Xs_elo.shape[1], k=F.K))):
            lv, lt, ep = [], [], []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                net, h = fit_torch(make(), (*pk(tr), tt(y[tr])), (*pk(va), tt(y[va])), seed=seed, lr=1e-3 if "trans" in name else 2e-3)
                ep.append(h["best_epoch"])
                lv.append(predict_logits(net, pk(va))); lt.append(predict_logits(net, pk(te)))
                if "trans" in name and seed == 0:
                    last_trans = (net, pk(te)[0])
            T = fit_temperature(np.mean(lv, 0), y[va])
            add(name, sig(np.mean(lt, 0) / T))
            info["folds"].setdefault(str(s), {})[name] = {"best_epochs": ep, "T": T}

        fl = {k: M.log_loss(v[-1], y[te]) for k, v in probs.items()}
        print(f"fold {s}: train {len(tr):,} val {len(va):,} test {len(te):,} [{time.time() - tf:.0f}s]  "
              f"bench {fl[BENCH]:.4f} control {fl[CONTROL]:.4f} logreg {fl['logreg[elo]']:.4f} "
              f"ens3 {fl[ENS]:.4f} gru {fl['gru[elo]']:.4f} trans {fl['transformer[elo]']:.4f}", flush=True)

    P = {k: np.concatenate(v) for k, v in probs.items()}
    Y = np.concatenate(ys)
    S = np.concatenate(seasons)
    res = {k: M.summarize(v, Y) for k, v in P.items()}
    for k in P:
        d, lo, hi = M.paired_bootstrap_delta(P[k], P[BENCH], Y)
        res[k].update({"dlogloss_vs_bench": d, "ci_lo": lo, "ci_hi": hi})
    print(f"\n=== POOLED WALK-FORWARD, test seasons {S.min()}-{S.max()} ({len(Y):,} games, home win rate {Y.mean():.3f}) ===")
    print(f"{'model':40s} {'logloss':>8s} {'brier':>7s} {'acc':>6s} {'ece':>6s}  vs own Elo [95% CI]")
    for k, r in sorted(res.items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{k:40s} {r['log_loss']:8.4f} {r['brier']:7.4f} {r['accuracy']:6.3f} {r['ece']:6.3f}"
              f"  {r['dlogloss_vs_bench']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")

    pairs = [("H1 style vs no style (logreg)", "logreg[elo+style]", "logreg[elo]"),
             ("H1 style vs no style (mlp)", "mlp[elo+style]", "mlp[elo]"),
             ("H1 style vs no style (hist_gb)", "hist_gb[elo+style]", "hist_gb[elo]"),
             ("H2 transformer vs GRU", "transformer[elo]", "gru[elo]"),
             ("H2 transformer vs logreg", "transformer[elo]", "logreg[elo]"),
             ("H2 GRU vs logreg", "gru[elo]", "logreg[elo]"),
             ("H3 ensemble3 vs control", ENS, CONTROL),
             ("H3 ensemble3 vs logreg", ENS, "logreg[elo]"),
             ("control vs own Elo", CONTROL, BENCH)]
    cmp = {}
    print("\nPAIRED COMPARISONS (log loss difference, negative = first is better):")
    for label, a, b in pairs:
        d, lo, hi = M.paired_bootstrap_delta(P[a], P[b], Y)
        cmp[label] = {"delta": d, "ci": [lo, hi]}
        print(f"  {label:34s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    per = pd.DataFrame({"season": S, "y": Y, **{k: v for k, v in P.items()}})
    rows = []
    for s, g in per.groupby("season"):
        rows.append({"season": int(s), "n": len(g), **{k: M.log_loss(g[k].to_numpy(), g["y"].to_numpy()) for k in P}})
    pd.DataFrame(rows).to_csv(OUT / "per_season_logloss.csv", index=False)

    # ---- figures ----
    names = sorted(res, key=lambda k: res[k]["log_loss"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ll = [res[k]["log_loss"] for k in names]
    col = ["#c0392b" if k == BENCH else "#1f6f8b" if k == ENS else "#9aa5b1" for k in names]
    ax.barh(names, ll, color=col); ax.invert_yaxis(); ax.set_xlim(min(ll) - 0.01, max(ll) + 0.004)
    for i, v in enumerate(ll):
        ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("pooled test log loss, walk-forward 2017-2024 (lower is better)")
    fig.tight_layout(); fig.savefig(OUT / "fig_wf_logloss.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    pdf = pd.DataFrame(rows)
    for k, c, lab in ((ENS, "#1f6f8b", "ensemble3"), (CONTROL, "#e67e22", "control (2 features)"), ("transformer[elo]", "#8e44ad", "transformer")):
        ax.plot(pdf.season, pdf[k] - pdf[BENCH], "o-", color=c, label=lab)
    ax.axhline(0, color="#555", lw=1)
    ax.set_ylabel("log loss minus own-Elo log loss"); ax.set_xlabel("test season"); ax.legend()
    ax.set_title("Per-season edge over raw Elo (negative = better)")
    fig.tight_layout(); fig.savefig(OUT / "fig_wf_by_season.png", dpi=160); plt.close(fig)

    if last_trans is not None:
        net, seq = last_trans
        att = net.cls_attention(seq).mean(0).numpy()
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.bar(range(1, F.K + 1), att[::-1], color="#8e44ad")
        ax.set_xlabel("games ago (1 = most recent)"); ax.set_ylabel("mean attention from [CLS]")
        ax.set_title(f"What the transformer reads: home team, {TEST_SEASONS[-1]} games")
        fig.tight_layout(); fig.savefig(OUT / "fig_wf_attention.png", dpi=160); plt.close(fig)
        info["attention_by_games_ago"] = att[::-1].tolist()

    (OUT / "walk_forward_results.json").write_text(json.dumps(
        {"n_games": int(len(Y)), "seasons": [int(S.min()), int(S.max())], "results": res, "comparisons": cmp, "info": info},
        indent=2, default=float))
    print(f"\nwrote {OUT}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
