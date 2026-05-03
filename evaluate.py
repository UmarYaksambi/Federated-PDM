"""
evaluate.py
===========
All evaluation logic in one place.

Functions:
  evaluate()      — RMSE, MAE, PHM score on a DataLoader
  mc_predict()    — MC-Dropout uncertainty estimation
  compute_metrics() — dict of all metrics from arrays

Called by every experiment script. Never duplicated elsewhere.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.loss import PHMScore


# Dataset wrapper

class CMAPSSDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, T, F)  float32
        self.y = torch.from_numpy(y)   # (N,)        float32

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        CMAPSSDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# Metrics

def compute_metrics(
    rul_pred: np.ndarray,
    rul_true: np.ndarray,
) -> dict:
    """
    Compute all metrics reported in the paper.

    Args:
        rul_pred: (N,) predicted RUL
        rul_true: (N,) ground-truth RUL

    Returns dict with keys:
        rmse, mae, mape, phm_score
    """
    err  = rul_pred - rul_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    # MAPE: guard against zero true values
    nonzero = rul_true != 0
    mape = float(
        np.mean(np.abs(err[nonzero] / rul_true[nonzero])) * 100
    ) if nonzero.any() else float("nan")

    # PHM asymmetric score
    phm = float(np.where(
        err < 0,
        np.exp(-err / 13.0) - 1,
        np.exp( err / 10.0) - 1,
    ).sum())

    return {"rmse": rmse, "mae": mae, "mape": mape, "phm_score": phm}


# Standard (deterministic) evaluation

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
) -> dict:
    """
    Deterministic evaluation: model in eval() mode (dropout off).
    Returns metrics dict.
    """
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return compute_metrics(preds, trues)


# MC-Dropout uncertainty estimation

def mc_predict(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    n_samples:  int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC-Dropout inference: run n_samples stochastic forward passes.

    Keep model in TRAIN mode so dropout remains active.
    Collect distribution of predictions per sample → mean + std.

    Returns:
        mean_pred:  (N,)  — point estimate (use for RMSE/MAE)
        std_pred:   (N,)  — epistemic uncertainty
        true_rul:   (N,)  — ground truth
    """
    model.train()   # dropout stays ON
    all_runs = []   # list of (n_samples,) arrays, one per sample

    # Collect all X, y first to avoid multi-pass DataLoader complexity
    all_X, all_y = [], []
    with torch.no_grad():
        for X, y in loader:
            all_X.append(X)
            all_y.append(y)
    all_X = torch.cat(all_X, dim=0).to(device)  # (N, T, F)
    all_y = torch.cat(all_y, dim=0).numpy()       # (N,)

    # n_samples stochastic passes
    run_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(all_X)               # (N,)
            run_preds.append(pred.cpu().numpy())

    run_preds  = np.stack(run_preds, axis=0)  # (n_samples, N)
    mean_pred  = run_preds.mean(axis=0)        # (N,)
    std_pred   = run_preds.std(axis=0)         # (N,)

    model.eval()   # restore eval mode
    return mean_pred, std_pred, all_y


# Results logging

def log_results(
    results: dict,
    csv_path: str,
    mode: str = "a",
):
    """Append a results dict as one row to a CSV file."""
    import csv, os
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)