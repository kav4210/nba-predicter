"""Training loop, prediction, and temperature scaling for the PyTorch models."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model: nn.Module, inputs: tuple[torch.Tensor, ...], bs: int = 4096) -> np.ndarray:
    model.eval()
    n = len(inputs[0])
    out = [model(*[t[i:i + bs] for t in inputs]) for i in range(0, n, bs)]
    return torch.cat(out).cpu().numpy()


def fit_torch(model, train, val, *, epochs=60, lr=2e-3, weight_decay=1e-2, batch_size=512,
              patience=8, seed=0):
    """Train with AdamW + cosine decay; early-stop on validation log loss.

    `train`/`val` are tuples (*inputs, y). Returns (model with best weights, history).
    """
    set_seed(seed)
    *x_tr, y_tr = train
    *x_va, y_va = val
    loader = DataLoader(TensorDataset(*x_tr, y_tr), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for *xb, yb in loader:
            opt.zero_grad()
            loss = bce(model(*xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        model.eval()
        with torch.no_grad():
            v = bce(torch.from_numpy(predict_logits(model, tuple(x_va))), y_va).item()
        hist["train"].append(tot / n)
        hist["val"].append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_val"] = best
    hist["best_epoch"] = int(np.argmin(hist["val"])) + 1
    return model, hist


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter calibration: find T minimising log loss of sigmoid(logit / T)."""
    z = torch.from_numpy(logits).float()
    t = torch.from_numpy(y).float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
