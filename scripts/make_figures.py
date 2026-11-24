from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nbapred import metrics as M  # noqa: E402

HEAD = "ENSEMBLE[hist_gb+mlp+seqnet, elo]"


def make_figures(results, info, probs, y_test, out, bench="538_elo_forecast"):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. test log loss with paired-bootstrap CI vs 538
    r = results["test"]
    names = sorted(r, key=lambda k: r[k]["log_loss"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ll = [r[k]["log_loss"] for k in names]
    colors = ["#c0392b" if k == bench else "#1f6f8b" if k == HEAD else "#9aa5b1" for k in names]
    ax.barh(names, ll, color=colors)
    ax.set_xlim(min(ll) - 0.01, max(ll) + 0.005)
    ax.invert_yaxis()
    ax.set_xlabel("test log loss (lower is better)")
    ax.set_title("Test seasons: log loss by model")
    for i, v in enumerate(ll):
        ax.text(v + 0.0004, i, f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_test_logloss.png", dpi=160)
    plt.close(fig)

    # 2. reliability diagram: headline vs 538
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1)
    for k, c, lab in ((HEAD, "#1f6f8b", "ensemble"), (bench, "#c0392b", "Elo benchmark")):
        rows = M.reliability(probs["test"][k], y_test, bins=8)
        ax.plot([a for a, _, _ in rows], [b for _, b, _ in rows], "o-", color=c, label=lab)
    ax.set_xlabel("predicted P(home win)")
    ax.set_ylabel("observed home win rate")
    ax.set_title("Calibration on test seasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_calibration.png", dpi=160)
    plt.close(fig)

    # 3. training curves (seed 0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, k in zip(axes, ("mlp[elo]", "seqnet[elo]")):
        h = info[k]["history_seed0"]
        ax.plot(h["train"], label="train")
        ax.plot(h["val"], label="validation")
        ax.axvline(info[k]["best_epochs"][0] - 1, color="#999", ls=":", lw=1)
        ax.set_title(k)
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_training_curves.png", dpi=160)
    plt.close(fig)

    # 4. permutation importance (validation)
    pi = info["hgb_perm_importance_val"][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([n for n, _ in pi], [v for _, v in pi], color="#1f6f8b")
    ax.set_xlabel("increase in validation log loss when shuffled")
    ax.set_title("Gradient boosting: permutation importance")
    fig.tight_layout()
    fig.savefig(out / "fig_importance.png", dpi=160)
    plt.close(fig)
