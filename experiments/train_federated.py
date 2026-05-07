"""
experiments/train_federated.py
================================
Runs all three federated experiments via --mode flag.

  E2: python experiments/train_federated.py --mode fedavg
  E3: python experiments/train_federated.py --mode fedprox
  E4: python experiments/train_federated.py --mode proposed

For all 5 seeds:
  python experiments/train_federated.py --mode proposed --all_seeds
"""

import argparse
import csv
import multiprocessing
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
import flwr as fl
from flwr.common import parameters_to_ndarrays
from torch.amp import GradScaler, autocast          # exposed for client code

from evaluate import compute_metrics, make_loader, mc_predict, log_results
from federation.client import make_client_fn
from federation.server import build_strategy
from models.tcn import build_model

# Number of DataLoader workers
NUM_WORKERS: int = min(8, multiprocessing.cpu_count())


# ------------------------------------------------------------------ #
# CUDA backend flags
# ------------------------------------------------------------------ #

def configure_cuda() -> None:
    """
    Enable kernel auto-tuning and TF32 math globally for this process.

    cudnn.benchmark  – selects the fastest conv kernel for your input shape.
    allow_tf32       – uses TensorFloat-32 for matmuls/convs on Ampere+ GPUs.
                       ~2× throughput vs strict FP32 with negligible RUL error.
    """
    torch.backends.cudnn.benchmark          = True
    torch.backends.cuda.matmul.allow_tf32   = True
    torch.backends.cudnn.allow_tf32         = True


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Mode → strategy config
# (use_simulation, use_fedprox, strategy_name)
MODE_CONFIG = {
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (True,  False, "similarity_weighted"),
}


# ------------------------------------------------------------------ #
# Strategy decorator
# ------------------------------------------------------------------ #

class _TrackingStrategy(fl.server.strategy.Strategy):
    """
    Decorator pattern wrapper around any Flower strategy.

    Captures:
      - per-round eval metrics → written to convergence CSV
      - final aggregated parameters → used for proper post-training eval

    Using Decorator rather than monkey-patching __class__ avoids
    MRO issues and broken isinstance() checks.
    """

    def __init__(self, wrapped: fl.server.strategy.Strategy):
        self._w = wrapped
        self.round_metrics: list[dict] = []
        self.last_params:   list[np.ndarray] | None = None

    # Delegation
    def initialize_parameters(self, client_manager):
        return self._w.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._w.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self._w.configure_evaluate(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self._w.evaluate(server_round, parameters)

    # Overrides with tracking
    def aggregate_fit(self, server_round, results, failures):
        params, metrics = self._w.aggregate_fit(server_round, results, failures)
        # Capture final global parameters after every round
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = self._w.aggregate_evaluate(server_round, results, failures)
        if metrics:
            record = dict(metrics)
            record["round"] = server_round
            self.round_metrics.append(record)
            if server_round % 10 == 0 or server_round == 1:
                kv = "  ".join(
                    f"{k}={v:.4f}" for k, v in record.items() if k != "round"
                )
                # cfg["federation"]["num_rounds"] — NOT top-level cfg["num_rounds"]
                if hasattr(self._w, "cfg"):
                    total_rounds = self._w.cfg.get("federation", {}).get(
                        "num_rounds", "?"
                    )
                else:
                    total_rounds = "?"
                print(f"  Round {server_round:>3}/{total_rounds}  {kv}")
        return loss, metrics


# ------------------------------------------------------------------ #
# Load final FL model
# ------------------------------------------------------------------ #

def _load_final_model(
    cfg:    dict,
    params: list[np.ndarray],
    device: torch.device,
) -> torch.nn.Module:
    """
    Construct a TCN and load the final aggregated FL parameters into it.

    Tensors are placed directly on `device` with non_blocking=True so the
    H2D transfer is overlapped with any subsequent CPU work.

    NOTE: `params` must be built from state_dict().values() (which includes
    both trainable parameters *and* buffers), matching how client.py's
    get_parameters() serialises the model.  Using model.parameters() on
    either side would silently drop buffers and misalign the mapping.
    """
    model = build_model(cfg).to(device)
    state_dict = model.state_dict()

    if len(params) != len(state_dict):
        raise ValueError(
            f"Parameter count mismatch: model state_dict has {len(state_dict)} "
            f"tensors but received {len(params)} from FL training. "
            f"Ensure get_parameters() uses state_dict().values(), not "
            f"model.parameters()."
        )

    new_state = {
        k: torch.tensor(v, dtype=torch.float32).to(device, non_blocking=True)
        for k, v in zip(state_dict.keys(), params)
    }
    model.load_state_dict(new_state, strict=True)
    return model


# ------------------------------------------------------------------ #
# Main training + evaluation
# ------------------------------------------------------------------ #

def run(cfg: dict, mode: str, seed: int) -> dict:
    set_seeds(seed)
    configure_cuda()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    use_simulation, use_fedprox, strategy_name = MODE_CONFIG[mode]
    num_clients = len(cfg["data"]["clients"])
    num_rounds  = cfg["federation"]["num_rounds"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(
        f"\n[Federated | {mode.upper()}]  seed={seed}  rounds={num_rounds}  "
        f"device={device}  AMP={use_amp}  workers={NUM_WORKERS}  "
        f"sim={'on' if use_simulation else 'off'}  "
        f"fedprox={'on' if use_fedprox else 'off'}"
    )

    # Build initial model + strategy
    init_model = build_model(cfg)

    # FIX 3: use state_dict().values() to serialise initial params.
    #
    # model.parameters() only yields trainable tensors and excludes buffers
    # (e.g. BatchNorm running stats, weight_norm pre-hook tensors).
    # state_dict().values() includes ALL tensors, matching what client.py's
    # get_parameters() / set_parameters() and _load_final_model() expect.
    # Mismatching the two sides silently corrupts buffer values every round.
    init_params = [val.detach().cpu().numpy()
                   for val in init_model.state_dict().values()]

    base_strategy = build_strategy(strategy_name, cfg, init_params)
    strategy      = _TrackingStrategy(base_strategy)

    # Client factory
    client_fn = make_client_fn(
        cfg,
        use_simulation,
        use_fedprox,
        device,
    )

    # GPU allocation for Ray virtual client engine (VCE).
    #
    # On Windows, Ray cannot manage CUDA resources at all.  Passing any
    # non-zero num_gpus causes Ray worker registration to fail with:
    #   core_worker_process.cc: Failed to register worker to Raylet:
    #   IOError: Unknown error
    # This is a known Ray limitation on Windows — CUDA context sharing via
    # Ray's resource scheduler only works on Linux/macOS.
    #
    # The correct Windows approach is num_gpus=0 so Ray does zero GPU
    # bookkeeping.  Each PDMClient receives the `device` object
    # (torch.device("cuda")) and moves tensors there with .to(device) —
    # PyTorch manages GPU access independently of Ray's resource tracking,
    # so training still runs fully on the RTX 3050.
    #
    # On Linux with a single GPU use 1.0/num_clients (fractional scheduling);
    # with multiple GPUs use 1.0 to give each virtual client an exclusive GPU.
    if sys.platform == "win32":
        gpu_fraction = 0.0   # Ray GPU management unsupported on Windows
    elif torch.cuda.is_available():
        gpu_fraction = 1.0 / num_clients   # fractional share on Linux/macOS
    else:
        gpu_fraction = 0.0

    if sys.platform == "win32":
        gpu_note = "Ray GPU mgmt disabled on Windows; PyTorch uses CUDA directly"
    else:
        gpu_note = f"1/{num_clients} GPU share per virtual client"
    print(f"  Client resources: cpus=2  gpus={gpu_fraction:.4f}  ({gpu_note})")

    # Federated simulation
    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = num_clients,
        config           = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy         = strategy,
        client_resources = {
            "num_cpus": 2,
            "num_gpus": gpu_fraction,
        },
    )

    # Sanity check
    if strategy.last_params is None:
        raise RuntimeError(
            "No aggregated parameters captured. "
            "Check that aggregate_fit() is being called and returning non-None."
        )

    # Load actual trained global model
    final_model = _load_final_model(cfg, strategy.last_params, device)

    torch.cuda.empty_cache()
    
    # Per-client final evaluation with MC-Dropout
    print("\n  Final per-client evaluation (MC-Dropout on trained global model):")
    results_row  = {"experiment": mode, "seed": seed}
    all_preds, all_trues = [], []

    for fd in cfg["data"]["clients"]:
        npz    = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        X_test = npz["X_test"]
        y_test = npz["y_test"]

        loader = make_loader(
            X_test,
            y_test,
            cfg["training"]["batch_size"],
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

        mean_pred, std_pred, true_rul = mc_predict(
            final_model, loader, device,
            n_samples=cfg["model"]["mc_samples"],
        )
        m = compute_metrics(mean_pred, true_rul)
        all_preds.append(mean_pred)
        all_trues.append(true_rul)

        print(f"    {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
              f"PHM={m['phm_score']:.1f}  Uncert±={std_pred.mean():.3f}")
        results_row[f"{fd}_rmse"] = round(m["rmse"],      4)
        results_row[f"{fd}_mae"]  = round(m["mae"],       4)
        results_row[f"{fd}_phm"]  = round(m["phm_score"], 2)
        results_row[f"{fd}_unc"]  = round(float(std_pred.mean()), 4)

    # Overall metrics
    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results_row["overall_rmse"] = round(overall["rmse"],      4)
    results_row["overall_mae"]  = round(overall["mae"],       4)
    results_row["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    # Save trained model checkpoint
    ckpt_path = os.path.join(results_dir, f"{mode}_seed{seed}.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # Save per-round convergence metrics
    if strategy.round_metrics:
        conv_path = os.path.join(results_dir, f"{mode}_convergence_seed{seed}.csv")
        with open(conv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=strategy.round_metrics[0].keys())
            writer.writeheader()
            writer.writerows(strategy.round_metrics)
        print(f"  Convergence: {conv_path}")

    # Append results row to CSV
    csv_path = os.path.join(results_dir, f"{mode}.csv")
    log_results(results_row, csv_path)
    print(f"  Results: {csv_path}")

    return results_row


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Federated PdM experiments (E2=fedavg, E3=fedprox, E4=proposed)"
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--mode",      choices=["fedavg", "fedprox", "proposed"],
                        required=True, help="Which experiment to run")
    parser.add_argument("--seed",      type=int, default=None,
                        help="Single seed. Omit to use config default.")
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all 5 seeds from config (for statistical significance).")
    parser.add_argument("--rounds",    type=int, default=None,
                        help="Override number of FL rounds (useful for quick testing).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.rounds:
        cfg["federation"]["num_rounds"] = args.rounds

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg["evaluation"]["seeds"] if args.all_seeds else [
            cfg["reproducibility"]["seed"]
        ]

    print(f"Experiment: {args.mode.upper()}  |  Seeds: {seeds}")
    for seed in seeds:
        run(cfg, args.mode, seed)

    print(f"\nDone. Results in ./results/{args.mode}.csv")


if __name__ == "__main__":
    main()