"""
experiments/train_federated.py
================================
Runs all three federated experiments via --mode flag.

  E2: python experiments/train_federated.py --mode fedavg
  E3: python experiments/train_federated.py --mode fedprox
  E4: python experiments/train_federated.py --mode proposed

For all 5 seeds:
  python experiments/train_federated.py --mode proposed --all_seeds \\
      --rounds 50 --local-epochs 5
"""

import argparse
import csv
import multiprocessing
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import yaml
import flwr as fl
from flwr.common import parameters_to_ndarrays

from evaluate import compute_metrics, make_loader, mc_predict, log_results
from federation.client import make_client_fn
from federation.server import build_strategy
from models.tcn import build_model

# DataLoader worker count — keep at 0 on Windows to avoid spawn overhead
if sys.platform == "win32":
    NUM_WORKERS: int = 0
else:
    NUM_WORKERS: int = min(4, multiprocessing.cpu_count())


# ------------------------------------------------------------------ #
# CUDA backend flags
# ------------------------------------------------------------------ #

def configure_cuda() -> None:
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True


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


# Mode → (use_simulation, use_fedprox, strategy_name)
MODE_CONFIG = {
    "fedavg":   (False, False, "fedavg"),
    "fedprox":  (False, True,  "fedprox"),
    "proposed": (True,  False, "similarity_weighted"),
}


# ------------------------------------------------------------------ #
# Tracking strategy wrapper
# ------------------------------------------------------------------ #

class _TrackingStrategy(fl.server.strategy.Strategy):
    """
    Decorator around any Flower strategy that:
      1. Captures per-round eval metrics → convergence CSV
      2. Captures final aggregated parameters → post-training eval
      3. Skips client evaluation when server_round % eval_every != 0
         (controlled by the eval_every constructor arg).

    Skipping evaluation is the largest single wall-clock saving in FL:
    a full eval round visits every client, runs inference on its test
    split, and returns metrics — nearly as expensive as a fit round.
    With eval_every=5 we save ~80 % of eval overhead with almost no
    loss in convergence visibility.
    """

    def __init__(
        self,
        wrapped:    fl.server.strategy.Strategy,
        eval_every: int = 5,
    ):
        self._w         = wrapped
        self.eval_every = max(1, eval_every)
        self.round_metrics: list[dict]          = []
        self.last_params:   list[np.ndarray] | None = None

    # ---- Delegation ----

    def initialize_parameters(self, client_manager):
        return self._w.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self._w.configure_fit(server_round, parameters, client_manager)

    def evaluate(self, server_round, parameters):
        return self._w.evaluate(server_round, parameters)

    # ---- Eval scheduling ----

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Return an empty list to skip evaluation on non-eval rounds."""
        if server_round % self.eval_every != 0:
            return []
        return self._w.configure_evaluate(server_round, parameters, client_manager)

    # ---- Tracked overrides ----

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = self._w.aggregate_fit(server_round, results, failures)
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = self._w.aggregate_evaluate(server_round, results, failures)

        # Free CUDA memory fragmented by Ray virtual-client teardown
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if metrics:
            record = dict(metrics)
            record["round"] = server_round
            self.round_metrics.append(record)

            total_rounds = (
                self._w.cfg.get("federation", {}).get("num_rounds", "?")
                if hasattr(self._w, "cfg") else "?"
            )
            kv = "  ".join(
                f"{k}={v:.4f}" for k, v in record.items() if k != "round"
            )
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
    model      = build_model(cfg).to(device)
    state_dict = model.state_dict()

    if len(params) != len(state_dict):
        raise ValueError(
            f"Parameter count mismatch: state_dict has {len(state_dict)} tensors "
            f"but received {len(params)} from FL training. "
            f"Ensure get_parameters() uses state_dict().values()."
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

def run(
    cfg:          dict,
    mode:         str,
    seed:         int,
    eval_every:   int = 5,
) -> dict:
    set_seeds(seed)
    configure_cuda()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cpu"

    use_simulation, use_fedprox, strategy_name = MODE_CONFIG[mode]
    num_clients = len(cfg["data"]["clients"])
    num_rounds  = cfg["federation"]["num_rounds"]
    local_ep    = cfg["training"]["epochs_local"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Effective training = rounds × local_epochs
    effective_ep = num_rounds * local_ep
    print(
        f"\n[Federated | {mode.upper()}]"
        f"  seed={seed}  rounds={num_rounds}  local_ep={local_ep}"
        f"  effective_ep={effective_ep}"
        f"  eval_every={eval_every}"
        f"  device={device}  AMP={use_amp}"
        f"  workers={NUM_WORKERS}"
        f"  sim={'on' if use_simulation else 'off'}"
        f"  fedprox={'on' if use_fedprox else 'off'}"
    )

    # Build initial model + serialise via state_dict (includes buffers)
    init_model  = build_model(cfg)
    init_params = [
        val.detach().cpu().numpy()
        for val in init_model.state_dict().values()
    ]

    base_strategy = build_strategy(strategy_name, cfg, init_params)
    strategy      = _TrackingStrategy(base_strategy, eval_every=eval_every)

    client_fn = make_client_fn(
        cfg,
        use_simulation,
        use_fedprox,
        device,
        use_amp=use_amp,
    )

    # GPU fraction for Ray VCE
    if sys.platform == "win32":
        gpu_fraction = 0.0
        gpu_note     = "Ray GPU mgmt disabled on Windows; PyTorch uses CUDA directly"
    elif torch.cuda.is_available():
        gpu_fraction = 1.0 / num_clients
        gpu_note     = f"1/{num_clients} GPU share per virtual client"
    else:
        gpu_fraction = 0.0
        gpu_note     = "CPU only"

    print(f"  Client resources: cpus=2  gpus={gpu_fraction:.4f}  ({gpu_note})")

    # ---- Federated simulation ----
    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = num_clients,
        config           = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy         = strategy,
        client_resources = {"num_cpus": 2, "num_gpus": gpu_fraction},
    )
    elapsed = time.time() - t0
    print(f"\n  Simulation finished in {elapsed/60:.1f} min")

    if strategy.last_params is None:
        raise RuntimeError(
            "No aggregated parameters captured. "
            "Check aggregate_fit() is returning non-None."
        )

    final_model = _load_final_model(cfg, strategy.last_params, device)
    torch.cuda.empty_cache()

    # ---- Per-client final evaluation (MC-Dropout) ----
    print("\n  Final per-client evaluation (MC-Dropout on trained global model):")
    results_row           = {"experiment": mode, "seed": seed}
    all_preds, all_trues  = [], []

    # pin_memory only safe when dataloader workers > 0
    use_pin = (NUM_WORKERS > 0) and (device.type == "cuda")

    for fd in cfg["data"]["clients"]:
        npz    = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        X_test = npz["X_test"]
        y_test = npz["y_test"]

        loader = make_loader(
            X_test,
            y_test,
            cfg["training"]["batch_size"],
            num_workers        = NUM_WORKERS,
            pin_memory         = use_pin,
            prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
            persistent_workers = NUM_WORKERS > 0,
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

    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results_row["overall_rmse"] = round(overall["rmse"],      4)
    results_row["overall_mae"]  = round(overall["mae"],       4)
    results_row["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"\n  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    # Save checkpoint
    ckpt_path = os.path.join(results_dir, f"{mode}_seed{seed}.pt")
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    # Convergence CSV
    if strategy.round_metrics:
        conv_path = os.path.join(
            results_dir, f"{mode}_convergence_seed{seed}.csv"
        )
        with open(conv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=strategy.round_metrics[0].keys()
            )
            writer.writeheader()
            writer.writerows(strategy.round_metrics)
        print(f"  Convergence: {conv_path}")

    # Append results row
    csv_path = os.path.join(results_dir, f"{mode}.csv")
    log_results(results_row, csv_path)
    print(f"  Results: {csv_path}")

    return results_row


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Federated PdM experiments (E2=fedavg, E3=fedprox, E4=proposed).\n"
            "\n"
            "Recommended settings for a fair comparison with centralised E1:\n"
            "  --rounds 50 --local-epochs 3 --eval-every 5\n"
            "\n"
            "This gives 150 effective local training epochs, comparable to\n"
            "centralised training at 50 epochs with full data access."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["fedavg", "fedprox", "proposed"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single seed. Omit to use config default.",
    )
    parser.add_argument(
        "--all_seeds",
        action="store_true",
        help="Run all 5 seeds from config (for statistical significance).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override number of FL rounds. Recommended: 50.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
        dest="local_epochs",
        help=(
            "Override cfg[training][epochs_local]. Each FL round runs this "
            "many local gradient steps per client. Recommended: 3–5."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        dest="eval_every",
        help=(
            "Run Flower evaluate step only every N rounds. "
            "Setting this to 5 saves ~80%% of eval overhead. "
            "Set to 1 to evaluate every round. Default: 5."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides to config dict before any client/strategy is created
    if args.rounds is not None:
        cfg["federation"]["num_rounds"] = args.rounds
    if args.local_epochs is not None:
        cfg["training"]["epochs_local"] = args.local_epochs
        print(
            f"[Override] epochs_local → {args.local_epochs}"
            f"  (effective training = {cfg['federation']['num_rounds']}"
            f" × {args.local_epochs} = "
            f"{cfg['federation']['num_rounds'] * args.local_epochs} steps)"
        )

    if args.seed is not None:
        seeds = [args.seed]
    elif args.all_seeds:
        seeds = cfg["evaluation"]["seeds"]
    else:
        seeds = [cfg["reproducibility"]["seed"]]

    print(f"Experiment: {args.mode.upper()}  |  Seeds: {seeds}")
    for seed in seeds:
        run(cfg, args.mode, seed, eval_every=args.eval_every)

    print(f"\nDone. Results in {cfg['evaluation']['results_dir']}/{args.mode}.csv")


if __name__ == "__main__":
    main()