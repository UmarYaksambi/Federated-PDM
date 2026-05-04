"""
evaluate.py
===========
All evaluation logic in one place — called by every experiment script.

Functions:
  make_loader()     — wraps numpy arrays into a DataLoader
  compute_metrics() — RMSE, MAE, MAPE, PHM score from numpy arrays
  evaluate()        — deterministic eval (dropout off) on a DataLoader
  mc_predict()      — MC-Dropout uncertainty estimation (N stochastic passes)
  log_results()     — append a results dict as one CSV row
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Dataset wrapper

class CMAPSSDataset(Dataset):
    """Wraps pre-windowed numpy arrays as a torch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, T, F)  float32
        self.y = torch.from_numpy(y)   # (N,)        float32

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool = False,
    num_workers: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    return DataLoader(
        CMAPSSDataset(X, y),
        batch_size = batch_size,
        shuffle    = shuffle,
        num_workers= num_workers,
        persistent_workers= persistent_workers,
        pin_memory = torch.cuda.is_available(),
    )


# Metrics

def compute_metrics(
    rul_pred: np.ndarray,
    rul_true: np.ndarray,
) -> dict:
    """
    Compute all metrics reported in the paper.

    Args:
        rul_pred: (N,) model predictions
        rul_true: (N,) ground-truth RUL

    Returns dict with keys:
        rmse, mae, mape, phm_score

    Note on PHM score:
        Asymmetric penalty — late predictions (positive error) are penalised
        more heavily than early predictions, reflecting safety priorities.
        S = Σ exp(-e/13)−1  if e < 0  (early)
          = Σ exp( e/10)−1  if e ≥ 0  (late)
    """
    err  = rul_pred - rul_true

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))

    # MAPE — guard against division by zero for RUL=0 samples
    nonzero = rul_true != 0
    mape = (
        float(np.mean(np.abs(err[nonzero] / rul_true[nonzero])) * 100)
        if nonzero.any() else float("nan")
    )

    # PHM asymmetric score (lower is better)
    phm = float(np.where(
        err < 0,
        np.exp(-err / 13.0) - 1.0,
        np.exp( err / 10.0) - 1.0,
    ).sum())

    return {"rmse": rmse, "mae": mae, "mape": mape, "phm_score": phm}


# Deterministic evaluation

@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Standard deterministic evaluation with dropout disabled.
    Returns a metrics dict from compute_metrics().
    """
    model.eval()
    preds, trues = [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds.append(model(X).cpu().numpy())
        trues.append(y.cpu().numpy())

    return compute_metrics(
        np.concatenate(preds),
        np.concatenate(trues),
    )


# MC-Dropout uncertainty estimation

def mc_predict(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC-Dropout inference: run n_samples stochastic forward passes with
    dropout active, then compute mean (point estimate) and std (uncertainty).

    Procedure:
      1. Set model.train() so Dropout layers remain active.
      2. Collect all test inputs in one pass (avoids re-iterating the loader).
      3. Run n_samples forward passes with torch.no_grad() for efficiency.
      4. Stack results → (n_samples, N) → compute per-sample statistics.
      5. Restore model.eval() before returning.

    Args:
        model:     trained TCN model
        loader:    test DataLoader
        device:    torch device
        n_samples: number of MC samples (50 is standard in literature)

    Returns:
        mean_pred: (N,) — point estimate (use for RMSE / MAE)
        std_pred:  (N,) — epistemic uncertainty
        true_rul:  (N,) — ground truth labels
    """
    model.train()   # dropout ON

    # Collect all inputs in one pass
    all_X, all_y = [], []
    with torch.no_grad():
        for X, y in loader:
            all_X.append(X)
            all_y.append(y)
    all_X    = torch.cat(all_X, dim=0).to(device)   # (N, T, F)
    true_rul = torch.cat(all_y, dim=0).numpy()        # (N,)

    # n_samples stochastic forward passes
    run_preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            run_preds.append(model(all_X).cpu().numpy())   # (N,)

    run_preds = np.stack(run_preds, axis=0)   # (n_samples, N)
    mean_pred = run_preds.mean(axis=0)         # (N,)
    std_pred  = run_preds.std(axis=0)          # (N,)

    model.eval()   # restore eval mode
    return mean_pred, std_pred, true_rul


# CSV logging

def log_results(
    results:  dict,
    csv_path: str,
    mode:     str = "a",
) -> None:
    """
    Append one results dict as a row to a CSV file.
    Writes header automatically if the file does not yet exist.
    """
    write_header = (
        not os.path.exists(csv_path)
        or os.path.getsize(csv_path) == 0
    )
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(results)