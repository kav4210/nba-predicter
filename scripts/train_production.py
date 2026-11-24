"""Fit the deployable ensemble (logreg + gradient boosting + MLP) on all completed games.

Validation = the most recent 1,230 completed games (about one season); training = everything
before. Validation picks logreg C, the number of trees, MLP early stopping, and the temperature.
Run after scripts/update_data.py so the model sees the newest games.

    python scripts/train_production.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbapred import features as F  # noqa: E402
from nbapred import metrics as M  # noqa: E402
from nbapred import modern  # noqa: E402
from nbapred.data import prepare_long  # noqa: E402
from nbapred.models import MLP  # noqa: E402
from nbapred.predict import MODELS_DIR  # noqa: E402
from nbapred.train import fit_temperature, fit_torch, predict_logits  # noqa: E402

N_VAL, SEEDS = 1230, 5
torch.set_num_threads(2)


def main():
    completed = modern.games_from_team_rows(modern.load_team_rows())
    games = F.build_games(F.add_team_features(prepare_long(modern.to_long(completed))))
    games = games[F.eligible(games)].reset_index(drop=True)
    cols = F.feature_columns(True)
    X = games[cols].to_numpy(dtype=np.float64)
    y = games["y"].to_numpy().astype(np.float32)
    tr, va = np.arange(len(games) - N_VAL), np.arange(len(games) - N_VAL, len(games))

    imp = SimpleImputer(strategy="median").fit(X[tr])
    sc = StandardScaler().fit(imp.transform(X[tr]))
    Xs = sc.transform(imp.transform(X)).astype(np.float32)

    best = None
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, max_iter=2000).fit(Xs[tr], y[tr])
        ll = M.log_loss(lr.predict_proba(Xs[va])[:, 1], y[va])
        if best is None or ll < best[0]:
            best = (ll, C, lr)
    logreg = best[2]

    kw = dict(learning_rate=0.04, max_depth=3, min_samples_leaf=100, l2_regularization=1.0,
              early_stopping=False, random_state=0)
    probe = HistGradientBoostingClassifier(max_iter=400, **kw).fit(X[tr], y[tr])
    n_trees = int(np.argmin([M.log_loss(p[:, 1], y[va]) for p in probe.staged_predict_proba(X[va])])) + 1
    hgb = HistGradientBoostingClassifier(max_iter=n_trees, **kw).fit(X[tr], y[tr])

    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    states, logits = [], []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        net, _ = fit_torch(MLP(Xs.shape[1]), (t(Xs[tr]), t(y[tr])), (t(Xs[va]), t(y[va])), seed=seed)
        states.append({k: v.clone() for k, v in net.state_dict().items()})
        logits.append(predict_logits(net, (t(Xs[va]),)))
    T = fit_temperature(np.mean(logits, axis=0), y[va])

    MODELS_DIR.mkdir(exist_ok=True)
    meta = {"trained_through": str(games["date"].max().date()), "n_train": int(len(tr)), "n_val": int(len(va)),
            "logreg_C": best[1], "n_trees": n_trees, "temperature": T}
    joblib.dump({"cols": cols, "imputer": imp, "scaler": sc, "logreg": logreg, "hgb": hgb,
                 "temperature": T, "meta": meta}, MODELS_DIR / "sklearn.joblib")
    torch.save({"states": states, "in_dim": Xs.shape[1]}, MODELS_DIR / "mlp.pt")

    # reference only: these validation games were used for tuning, so this is not a test score
    from nbapred.predict import load_bundle, predict_home_win  # noqa: E402
    p = predict_home_win(load_bundle(), games.iloc[va])
    meta["validation_logloss"] = M.log_loss(p["p_home"].to_numpy(), y[va])
    meta["validation_accuracy"] = M.accuracy(p["p_home"].to_numpy(), y[va])
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
