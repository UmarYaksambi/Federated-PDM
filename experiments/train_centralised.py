"""
experiments/train_centralised.py
=================================
Experiment E1 — Centralised Training (Upper-Bound Baseline)

All 4 client datasets pooled into one. Trained as a normal PyTorch model.
This gives the theoretical performance ceiling that federated methods aim for.

Usage:
    python experiments/train_centralised.py
    python experiments/train_centralised.py --seed 123
    python experiments/train_centralised.py --epochs 50
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluate import CMAPSSDataset, compute_metrics, evaluate, log_results, make_loader, mc_predict
from models.loss import HybridRULLoss, build_criterion
from models.tcn import build_model


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_all_clients(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Pool all client training data → single train set.
    Keep test sets separate (evaluate per-client and overall).
    """
    train_X, train_y = [], []
    test_data = {}

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


def train(cfg: dict, seed: int, epochs: int) -> dict:
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Centralised] seed={seed}  device={device}")

    # Data
    X_train, y_train, test_data = load_all_clients(cfg)
    train_loader = make_loader(X_train, y_train,
                               cfg["training"]["batch_size"], shuffle=True)
    print(f"  Train: {X_train.shape}  |  "
          f"Test clients: {list(test_data.keys())}")

    # Model
    model     = build_model(cfg).to(device)
    criterion = build_criterion(cfg)
    opt       = AdamW(model.parameters(),
                      lr=cfg["training"]["learning_rate"],
                      weight_decay=cfg["training"]["weight_decay"])
    sched     = CosineAnnealingLR(opt, T_max=epochs)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            rul_pred, hi_seq = model.forward_with_hi(X_b)
            loss, _ = criterion(rul_pred, y_b, hi_seq)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(train_loader)
            print(f"  Epoch {epoch:>3}/{epochs}  loss={avg:.4f}")

    # Evaluation per client
    print("\n  Per-client test results:")
    results = {"experiment": "centralised", "seed": seed}
    all_preds, all_trues = [], []

    for fd, (X_test, y_test) in test_data.items():
        loader = make_loader(X_test, y_test, cfg["training"]["batch_size"])

        # Standard metrics
        m = evaluate(model, loader, device)

        # MC-Dropout uncertainty
        mean_pred, std_pred, true_rul = mc_predict(
            model, loader, device, n_samples=cfg["model"]["mc_samples"]
        )
        all_preds.append(mean_pred)
        all_trues.append(true_rul)

        print(f"    {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
              f"PHM={m['phm_score']:.1f}  UncertMean={std_pred.mean():.2f}")
        results[f"{fd}_rmse"] = round(m["rmse"], 4)
        results[f"{fd}_mae"]  = round(m["mae"],  4)
        results[f"{fd}_phm"]  = round(m["phm_score"], 2)

    # Overall metrics
    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results["overall_rmse"] = round(overall["rmse"], 4)
    results["overall_mae"]  = round(overall["mae"],  4)
    results["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    # Save checkpoint
    os.makedirs(cfg["evaluation"]["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(
        cfg["evaluation"]["results_dir"], f"centralised_seed{seed}.pt"
    )
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Saved model: {ckpt_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Single seed. If omitted, runs all 5 seeds from config.")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seeds  = [args.seed] if args.seed else cfg["evaluation"]["seeds"]
    epochs = args.epochs or cfg["training"]["epochs_local"] * cfg["federation"]["num_rounds"]

    csv_path = os.path.join(cfg["evaluation"]["results_dir"], "centralised.csv")
    print(f"Running Experiment E1 (Centralised) — {len(seeds)} seed(s)")

    for seed in seeds:
        results = train(cfg, seed, epochs)
        log_results(results, csv_path)

    print(f"\nAll results saved to {csv_path}")


if __name__ == "__main__":
    main()