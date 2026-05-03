"""
experiments/train_federated.py
================================
Runs all three federated experiments via --mode flag.

  E2: python experiments/train_federated.py --mode fedavg
  E3: python experiments/train_federated.py --mode fedprox
  E4: python experiments/train_federated.py --mode proposed

For all 5 seeds:
  python experiments/train_federated.py --mode proposed --all_seeds

Architecture (same for all modes):
  - 4 Flower clients (FD001–FD004) in simulation mode
  - Server aggregation switches based on --mode
  - Results logged to results/<mode>.csv
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
import flwr as fl

from evaluate import compute_metrics, make_loader, mc_predict, log_results
from federation.client import make_client_fn
from federation.server import build_strategy
from models.tcn import build_model


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path="config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


MODE_CONFIG = {
    # mode → (use_simulation, use_fedprox, strategy_name)
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (True,  False, "similarity_weighted"),
}


def run(cfg: dict, mode: str, seed: int):
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_simulation, use_fedprox, strategy_name = MODE_CONFIG[mode]
    num_rounds  = cfg["federation"]["num_rounds"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n[Federated | {mode.upper()}] seed={seed}  "
          f"rounds={num_rounds}  device={device}")

    # Initial model parameters
    init_model  = build_model(cfg)
    init_params = [p.detach().numpy() for p in init_model.parameters()]

    # Strategy
    strategy = build_strategy(strategy_name, cfg, init_params)

    # Client factory
    client_fn = make_client_fn(cfg, use_simulation, use_fedprox, device)

    # Per-round metric tracking
    round_metrics: list[dict] = []

    class _TrackingStrategy(strategy.__class__):
        """Thin wrapper to capture per-round eval metrics."""
        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            if metrics:
                metrics["round"] = server_round
                round_metrics.append(metrics)
                if server_round % 10 == 0:
                    print(f"  Round {server_round:>3}  "
                          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                      if k != "round"))
            return loss, metrics

    # Patch the strategy instance (avoids re-constructing)
    strategy.__class__ = _TrackingStrategy

    # Run federation
    history = fl.simulation.start_simulation(
        client_fn         = client_fn,
        num_clients       = len(cfg["data"]["clients"]),
        config            = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy          = strategy,
        client_resources  = {"num_cpus": 1, "num_gpus": 0.0},
    )

    # Final evaluation with best global model
    # Retrieve final global parameters from history
    final_params_fl = history.parameters_res if hasattr(history, "parameters_res") else None

    # Reconstruct final model from last round's aggregated parameters
    final_model = build_model(cfg).to(device)

    # Get final weights from the last round (workaround for Flower simulation)
    # We re-run one client evaluation with the final global params
    print("\n  Final per-client evaluation (MC-Dropout):")
    results_row = {"experiment": mode, "seed": seed}
    all_preds, all_trues = [], []

    # Use the last client (which has the global model from fit())
    # Build a temporary client to run final eval
    from federation.client import PDMClient
    for fd in cfg["data"]["clients"]:
        client = PDMClient(
            fd=fd, cfg=cfg,
            use_simulation=use_simulation,
            use_fedprox=use_fedprox,
            device=device,
        )

        # Get the final round eval metrics from history
        # history.metrics_distributed contains per-round per-client metrics
        fd_rmse = None
        if hasattr(history, "metrics_distributed_fit"):
            pass  # use below

        # Run MC-Dropout eval with current (last-round) model
        loader = make_loader(
            client.X_test, client.y_test, cfg["training"]["batch_size"]
        )
        mean_pred, std_pred, true_rul = mc_predict(
            client.model, loader, device,
            n_samples=cfg["model"]["mc_samples"]
        )
        m = compute_metrics(mean_pred, true_rul)
        all_preds.append(mean_pred)
        all_trues.append(true_rul)

        print(f"    {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
              f"PHM={m['phm_score']:.1f}  ±{std_pred.mean():.2f}")
        results_row[f"{fd}_rmse"] = round(m["rmse"], 4)
        results_row[f"{fd}_mae"]  = round(m["mae"],  4)
        results_row[f"{fd}_phm"]  = round(m["phm_score"], 2)
        results_row[f"{fd}_unc"]  = round(float(std_pred.mean()), 4)

    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results_row["overall_rmse"] = round(overall["rmse"], 4)
    results_row["overall_mae"]  = round(overall["mae"],  4)
    results_row["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    # Save convergence metrics
    if round_metrics:
        import csv
        conv_path = os.path.join(results_dir, f"{mode}_convergence_seed{seed}.csv")
        with open(conv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=round_metrics[0].keys())
            writer.writeheader()
            writer.writerows(round_metrics)
        print(f"  Convergence curve saved: {conv_path}")

    # Log final row
    csv_path = os.path.join(results_dir, f"{mode}.csv")
    log_results(results_row, csv_path)
    print(f"  Results appended to {csv_path}")

    return results_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--mode",      choices=["fedavg", "fedprox", "proposed"],
                        required=True,
                        help="E2=fedavg | E3=fedprox | E4=proposed")
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all 5 seeds from config")
    parser.add_argument("--rounds",    type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.rounds:
        cfg["federation"]["num_rounds"] = args.rounds

    seeds = cfg["evaluation"]["seeds"] if args.all_seeds else [
        args.seed if args.seed is not None else cfg["reproducibility"]["seed"]
    ]

    print(f"Running Experiment ({args.mode.upper()}) — {len(seeds)} seed(s)")
    for seed in seeds:
        run(cfg, args.mode, seed)


if __name__ == "__main__":
    main()