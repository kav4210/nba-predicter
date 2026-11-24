"""Unsupervised: do NBA teams fall into playing-style archetypes, and are they real?

Unit of analysis = team-season (about 420 rows, 2011-2024). Features = how a team plays
(pace, 3-point rate, free-throw rate, turnovers, rebounding, assists, steals, blocks), not how
good it is. Each metric is z-scored WITHIN its season so clusters compare teams to their
contemporaries; league-wide drift (the 3-point revolution) is shown separately.

Questions answered, not assumed:
  1. How many clusters? (silhouette and GMM BIC; they may disagree)
  2. Are the clusters stable under resampling, or an artifact of k-means?
  3. Does cluster membership relate to winning, or is style mostly orthogonal to quality?

    python scripts/style_analysis.py
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import modern  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402
from nbapred.style import ARCHETYPE, game_metrics  # noqa: E402

OUT = ROOT / "results" / "style"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def team_seasons() -> pd.DataFrame:
    long = prepare_long(modern.to_long(modern.games_from_team_rows(modern.load_team_rows())))
    m = game_metrics(long)
    df = pd.concat([long[["fran_id", "team_id", "year_id", "win"]], m], axis=1)
    ts = df.groupby(["fran_id", "year_id"]).agg(
        team=("team_id", "last"), win_pct=("win", "mean"), games=("win", "size"),
        **{c: (c, "mean") for c in ARCHETYPE + ["ortg", "drtg", "efg"]},
    ).reset_index()
    return ts[ts.games >= 50].reset_index(drop=True)  # drop the shortened/odd partials


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts = team_seasons()
    z = ts.groupby("year_id")[ARCHETYPE].transform(lambda s: (s - s.mean()) / s.std())
    Z = z.to_numpy()
    print(f"{len(ts)} team-seasons, {ts.year_id.min()}-{ts.year_id.max()}")

    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    P = pca.transform(Z)[:, :2]
    loadings = pd.DataFrame(pca.components_[:2].T, index=ARCHETYPE, columns=["PC1", "PC2"])
    print("explained variance:", np.round(evr[:4], 3), "cumulative top 3 =", round(evr[:3].sum(), 3))
    print(loadings.round(2).to_string())

    # 1. how many clusters?
    ks = list(range(2, 9))
    sil = {k: float(silhouette_score(Z, KMeans(k, n_init=20, random_state=0).fit_predict(Z))) for k in ks}
    bic = {k: float(GaussianMixture(k, covariance_type="diag", n_init=5, random_state=0).fit(Z).bic(Z)) for k in ks}
    k = max(range(3, 9), key=lambda j: sil[j])  # at least 3: k=2 usually just splits on the biggest axis
    print("silhouette:", {j: round(v, 3) for j, v in sil.items()}, "-> k =", k)
    print("GMM BIC best k:", min(bic, key=bic.get))

    km = KMeans(k, n_init=50, random_state=0).fit(Z)
    ts["cluster"] = km.labels_

    # 2. stability: fit on 80%, assign the held-out 20%, compare with the full-data labels
    rng = np.random.default_rng(0)
    aris = []
    for _ in range(40):
        idx = rng.permutation(len(Z))
        tr, te = idx[: int(0.8 * len(Z))], idx[int(0.8 * len(Z)):]
        sub = KMeans(k, n_init=10, random_state=0).fit(Z[tr])
        aris.append(adjusted_rand_score(km.labels_[te], sub.predict(Z[te])))
    stab = float(np.mean(aris))
    print(f"stability (held-out ARI vs full fit): {stab:.2f} +/- {np.std(aris):.2f}  (1 = identical, 0 = chance)")

    # shuffled-data baseline: silhouette when each column is independently permuted (no structure)
    null = []
    for _ in range(20):
        Zp = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        null.append(silhouette_score(Zp, KMeans(k, n_init=10, random_state=0).fit_predict(Zp)))
    print(f"silhouette real {sil[k]:.3f} vs shuffled-columns null {np.mean(null):.3f} +/- {np.std(null):.3f}")

    # 3. profiles, and does style relate to winning?
    prof = pd.DataFrame(Z, columns=ARCHETYPE).groupby(ts["cluster"]).mean()
    names = {}
    for c, row in prof.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        names[int(c)] = " ".join(("+" if row[f] > 0 else "-") + f for f in top)
    summary = ts.groupby("cluster").agg(n=("team", "size"), win_pct=("win_pct", "mean"), ortg=("ortg", "mean"),
                                        drtg=("drtg", "mean")).round(3)
    summary["label"] = pd.Series(names)
    print(summary.to_string())
    from scipy.stats import f_oneway
    f, p = f_oneway(*[g.win_pct.to_numpy() for _, g in ts.groupby("cluster")])
    print(f"ANOVA win% across clusters: F={f:.2f}, p={p:.3f}")

    drift = ts.groupby("year_id")[["pace", "three_rate", "ft_rate", "tov_rate", "oreb_pct"]].mean()

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(P[:, 0], P[:, 1], c=ts.cluster, cmap="tab10", s=18, alpha=0.8)
    far = np.argsort(-np.hypot(P[:, 0], P[:, 1]))[:9]
    for i in far:
        ax.annotate(f"{ts.team[i]} {ts.year_id[i]}", P[i], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%}): " + ", ".join(loadings.PC1.abs().sort_values(ascending=False).index[:2]))
    ax.set_ylabel(f"PC2 ({evr[1]:.0%}): " + ", ".join(loadings.PC2.abs().sort_values(ascending=False).index[:2]))
    ax.set_title(f"Team-season playing styles (k={k}, within-season z-scores)")
    fig.tight_layout(); fig.savefig(OUT / "fig_style_pca.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 0.6 * k + 2))
    im = ax.imshow(prof.to_numpy(), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(ARCHETYPE)), ARCHETYPE, rotation=40, ha="right")
    ax.set_yticks(range(k), [f"{c}: {names[c]}  (n={int(summary.n[c])})" for c in range(k)])
    for i in range(k):
        for j in range(len(ARCHETYPE)):
            ax.text(j, i, f"{prof.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Cluster profiles (mean within-season z-score)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_style_profiles.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ks, [sil[j] for j in ks], "o-"); axes[0].axvline(k, ls=":", color="#999")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("silhouette"); axes[0].set_title("k-means silhouette")
    axes[1].plot(ks, [bic[j] for j in ks], "o-", color="#c0392b"); axes[1].set_xlabel("k")
    axes[1].set_ylabel("BIC (lower is better)"); axes[1].set_title("Gaussian mixture BIC")
    fig.tight_layout(); fig.savefig(OUT / "fig_style_k.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(drift.index, drift.three_rate, "o-", label="3PA / FGA")
    ax2 = ax.twinx(); ax2.plot(drift.index, drift.pace, "s--", color="#c0392b", label="pace")
    ax.set_ylabel("3-point attempt rate"); ax2.set_ylabel("possessions per game")
    ax.set_title("League-wide style drift (raw, not z-scored)")
    fig.tight_layout(); fig.savefig(OUT / "fig_style_drift.png", dpi=160); plt.close(fig)

    (OUT / "style_results.json").write_text(json.dumps({
        "n_team_seasons": len(ts), "explained_variance": evr[:5].tolist(),
        "loadings": loadings.round(3).to_dict(), "silhouette": sil, "gmm_bic": bic, "chosen_k": k,
        "stability_ari_mean": stab, "stability_ari_sd": float(np.std(aris)),
        "silhouette_null_mean": float(np.mean(null)), "anova_winpct_F": float(f), "anova_winpct_p": float(p),
        "clusters": summary.reset_index().to_dict(orient="records"),
        "three_rate_2011": float(drift.three_rate.iloc[0]), "three_rate_2024": float(drift.three_rate.iloc[-1]),
    }, indent=2, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
