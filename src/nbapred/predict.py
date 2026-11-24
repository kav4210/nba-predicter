"""Production path: features for upcoming games, and the saved ensemble that scores them.

The slate path reuses exactly the same code as the backtest (modern.to_long -> add_team_features
-> build_games), so an upcoming game gets the same features it would have had in a backtest.
tests/test_predict_path.py checks that equivalence.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import features as F
from . import modern
from .data import prepare_long
from .models import MLP

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def slate_features(completed: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Feature rows (one per upcoming game) built from all completed games before them."""
    allg = pd.concat([completed, upcoming], ignore_index=True)
    long_feat = F.add_team_features(prepare_long(modern.to_long(allg)))
    games = F.build_games(long_feat)
    return games[games["is_upcoming"]].reset_index(drop=True)


def load_bundle(models_dir: Path = MODELS_DIR) -> dict:
    b = joblib.load(models_dir / "sklearn.joblib")
    mlp = torch.load(models_dir / "mlp.pt", weights_only=True)
    nets = []
    for sd in mlp["states"]:
        net = MLP(mlp["in_dim"])
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    b["nets"] = nets
    return b


def predict_home_win(bundle: dict, feats: pd.DataFrame) -> pd.DataFrame:
    X = feats[bundle["cols"]].to_numpy(dtype=np.float64)
    Xs = bundle["scaler"].transform(bundle["imputer"].transform(X)).astype(np.float32)
    p_lr = bundle["logreg"].predict_proba(Xs)[:, 1]
    p_gb = bundle["hgb"].predict_proba(X)[:, 1]
    with torch.no_grad():
        z = np.mean([n(torch.from_numpy(Xs)).numpy() for n in bundle["nets"]], axis=0)
    p_mlp = 1 / (1 + np.exp(-z / bundle["temperature"]))
    return pd.DataFrame({"p_logreg": p_lr, "p_hist_gb": p_gb, "p_mlp": p_mlp,
                         "p_home": (p_lr + p_gb + p_mlp) / 3.0})
