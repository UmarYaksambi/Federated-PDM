"""
experiments/train_centralised.py
=================================
Experiment E1 — Centralised Training (Upper-Bound Baseline)

All 4 client datasets pooled into one training set.
This is the theoretical performance ceiling — a model trained on all data
with no federation overhead or Non-IID constraints.
Federated results should be compared against this ceiling.

Usage:
    python experiments/train_centralised.py                # all 5 seeds
    python experiments/train_centralised.py --seed 42      # single seed
    python experiments/train_centralised.py --all_seeds    # explicit all seeds
    python experiments/train_centralised.py --epochs 100   # override epoch count
"""

import argparse
import multiprocessing
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluate import compute_metrics, evaluate, log_results, make_loader, mc_predict
from models.loss import build_criterion
from models.tcn import build_model

# Number of DataLoader workers
# Capped at 8 to avoid excessive RAM pressure; tune down if OOM.
NUM_WORKERS: int = min(8, multiprocessing.cpu_count())


# CUDA backend flags
def configure_cuda() -> None:
    """
    Enable kernel auto-tuning and TF32 math.

    cudnn.benchmark  – profiles conv kernels on first batch and picks the
                       fastest implementation for your exact input shape.
                       Disable if input shapes vary wildly across batches.

    allow_tf32       – uses 10-bit mantissa (vs 23-bit FP32) for matmuls
                       and convolutions on Ampere+ GPUs (~2× throughput,
                       negligible accuracy loss for RUL regression).
    """
    torch.backends.cudnn.benchmark          = True
    torch.backends.cuda.matmul.allow_tf32   = True
    torch.backends.cudnn.allow_tf32         = True


# Reproducibility
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Data loading
def load_all_clients(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Pool all client training data into one centralised training set.
    Test sets are kept separate so per-client metrics can be reported.
    """
    train_X, train_y = [], []
    test_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for fd in cfg["data"]["clients"]:
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        train_X.append(npz["X_train"])
        train_y.append(npz["y_train"])
        test_data[fd] = (npz["X_test"], npz["y_test"])

    return (
        np.concatenate(train_X, axis=0),
        np.concatenate(train_y, axis=0),
        test_data,
    )


# Training
def train(cfg: dict, seed: int, epochs: int) -> dict:
    set_seeds(seed)
    configure_cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"          # AMP is only beneficial on GPU
    print(f"\n[Centralised]  seed={seed}  device={device}  "
          f"epochs={epochs}  AMP={use_amp}  workers={NUM_WORKERS}")

    # Data
    X_train, y_train, test_data = load_all_clients(cfg)
    train_loader = make_loader(
        X_train,
        y_train,
        cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,            # async page-locked CPU→GPU DMA
        prefetch_factor=2,          # prefetch 2 batches per worker
        persistent_workers=True,    # keep worker processes alive between epochs
    )
    print(f"  Train: {X_train.shape}  |  Clients: {list(test_data.keys())}")

    # Model + optimiser
    model     = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    opt       = AdamW(
        model.parameters(),
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    sched  = CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)    # loss scaler for stable FP16 grads

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for X_b, y_b in train_loader:
            # non_blocking=True overlaps H2D transfer with CPU work
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            opt.zero_grad()

            # FP16 forward pass under AMP context
            with autocast(device_type=device.type, enabled=use_amp):
                rul_pred, hi_seq = model.forward_with_hi(X_b)
                loss, _          = criterion(rul_pred, y_b, hi_seq)

            # Scaled backward + unscale before clipping
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()

        sched.step()

        if epoch == 1 or epoch % 10 == 0:
            avg = total_loss / max(len(train_loader), 1)
            print(f"  Epoch {epoch:>4}/{epochs}  loss={avg:.4f}")

    # Per-client evaluation
    print("\n  Per-client test results (deterministic + MC-Dropout):")
    results  = {"experiment": "centralised", "seed": seed}
    all_preds, all_trues = [], []

    for fd, (X_test, y_test) in test_data.items():
        loader = make_loader(
            X_test,
            y_test,
            cfg["training"]["batch_size"],
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        # Deterministic metrics
        m_det = evaluate(model, loader, device)

        # MC-Dropout uncertainty
        mean_pred, std_pred, true_rul = mc_predict(
            model, loader, device,
            n_samples=cfg["model"]["mc_samples"],
        )
        all_preds.append(mean_pred)
        all_trues.append(true_rul)

        print(f"    {fd}: RMSE={m_det['rmse']:.2f}  MAE={m_det['mae']:.2f}  "
              f"PHM={m_det['phm_score']:.1f}  Uncert±={std_pred.mean():.3f}")

        results[f"{fd}_rmse"] = round(m_det["rmse"],      4)
        results[f"{fd}_mae"]  = round(m_det["mae"],       4)
        results[f"{fd}_phm"]  = round(m_det["phm_score"], 2)
        results[f"{fd}_unc"]  = round(float(std_pred.mean()), 4)

    # Overall metrics
    overall = compute_metrics(
        np.concatenate(all_preds),
        np.concatenate(all_trues),
    )
    results["overall_rmse"] = round(overall["rmse"],      4)
    results["overall_mae"]  = round(overall["mae"],       4)
    results["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    # Save checkpoint
    os.makedirs(cfg["evaluation"]["results_dir"], exist_ok=True)
    ckpt = os.path.join(
        cfg["evaluation"]["results_dir"], f"centralised_seed{seed}.pt"
    )
    torch.save(model.state_dict(), ckpt)
    print(f"  Checkpoint: {ckpt}")

    return results


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="E1 — Centralised upper-bound baseline."
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Single seed. Omit to run all seeds from config.")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Explicitly run all 5 seeds (same as omitting --seed).")
    parser.add_argument("--epochs",    type=int, default=None,
                        help="Override total epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg["evaluation"]["seeds"]

    # Default epoch count: local_epochs × num_rounds (equivalent FL budget)
    epochs = args.epochs or (
        cfg["training"]["epochs_local"] * cfg["federation"]["num_rounds"]
    )

    csv_path = os.path.join(cfg["evaluation"]["results_dir"], "centralised.csv")
    os.makedirs(cfg["evaluation"]["results_dir"], exist_ok=True)

    print(f"Experiment E1 — Centralised  |  seeds={seeds}  epochs={epochs}")
    for seed in seeds:
        row = train(cfg, seed, epochs)
        log_results(row, csv_path)

    print(f"\nAll results saved → {csv_path}")


if __name__ == "__main__":
    main()