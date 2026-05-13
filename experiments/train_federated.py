"""
experiments/train_federated.py
================================
Runs all three federated experiments via --mode flag.

  E2: python experiments/train_federated.py --mode fedavg
  E3: python experiments/train_federated.py --mode fedprox
  E4: python experiments/train_federated.py --mode proposed

Recommended baseline run (verify FL is working):
  python experiments/train_federated.py --mode proposed \\
      --rounds 10 --local-epochs 1 --eval-every 2

Recommended full run for paper comparison vs E1 (50 centralised epochs):
  python experiments/train_federated.py --mode proposed \\
      --rounds 50 --local-epochs 3 --eval-every 5

For all 5 seeds:
  python experiments/train_federated.py --mode proposed --all_seeds \\
      --rounds 50 --local-epochs 3

CHANGES IN THIS VERSION
=======================
- --local-epochs default changed to 1 in argument help.
  With local_epochs=5 in the original run, clients drifted so far that
  RMSE got WORSE across rounds (46.98 → 60.49).  Use 1–2 for stability;
  3 for faster wall-clock convergence if monitoring confirms no divergence.

- --use-fedprox flag added.
  Allows enabling the FedProx proximal term in the "proposed" mode
  without switching to --mode fedprox (which uses plain FedAvg on the
  server).  This combination (similarity-weighted aggregation + proximal
  constraint) is the strongest defence against client drift.

- MODE_CONFIG updated to look up use_fedprox dynamically.
  Proposed mode still defaults to use_fedprox=False to match paper
  description; use --use-fedprox to enable it.

- eval_every=5 default (unchanged from previous version).
  Still the single biggest wall-clock saving.
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


# (use_simulation, strategy_name)
# use_fedprox is now set via CLI --use-fedprox, not locked in MODE_CONFIG,
# because the paper's "proposed" mode may or may not combine with FedProx.
_MODE_BASE = {
    "fedavg":   (False, "fedavg"),
    "fedprox":  (False, "fedprox"),   # proximal term from mode name
    "proposed": (True,  "similarity_weighted"),
}


# ------------------------------------------------------------------ #
# Tracking strategy wrapper
# ------------------------------------------------------------------ #

class _TrackingStrategy(fl.server.strategy.Strategy):
    """
    Decorator around any Flower strategy.
    Captures per-round eval metrics and final aggregated parameters.
    Skips evaluation on non-eval rounds (eval_every > 1) for speed.
    """

    def __init__(
        self,
        wrapped:    fl.server.strategy.Strategy,
        eval_every: int = 5,
    ):
        self._w         = wrapped
        self.eval_every = max(1, eval_every)
        self.round_metrics: list[dict]              = []
        self.last_params:   list[np.ndarray] | None = None

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self._w.initialize_parameters(client_manager=client_manager, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        return self._w.configure_fit(server_round=server_round, parameters=parameters, client_manager=client_manager, **kwargs)

    def evaluate(self, server_round, parameters, **kwargs):
        return self._w.evaluate(server_round=server_round, parameters=parameters, **kwargs)

    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
        if server_round % self.eval_every != 0:
            return []
        return self._w.configure_evaluate(server_round=server_round, parameters=parameters, client_manager=client_manager, **kwargs)

    def aggregate_fit(self, server_round, results, failures, **kwargs):
        params, metrics = self._w.aggregate_fit(server_round=server_round, results=results, failures=failures, **kwargs)
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures, **kwargs):
        loss, metrics = self._w.aggregate_evaluate(server_round=server_round, results=results, failures=failures, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if metrics:
            record        = dict(metrics)
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
            f"but received {len(params)}. "
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
    eval_every:   int  = 5,
    use_fedprox:  bool = False,
) -> dict:
    set_seeds(seed)
    configure_cuda()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    use_simulation, strategy_name = _MODE_BASE[mode]

    # --mode fedprox always enables the proximal term regardless of --use-fedprox
    # --mode proposed + --use-fedprox = similarity-weighted + proximal constraint
    _use_fedprox = use_fedprox or (mode == "fedprox")

    num_clients = len(cfg["data"]["clients"])
    num_rounds  = cfg["federation"]["num_rounds"]
    local_ep    = cfg["training"]["epochs_local"]
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    effective_ep = num_rounds * local_ep
    print(
        f"\n[Federated | {mode.upper()}]"
        f"  seed={seed}"
        f"  rounds={num_rounds}"
        f"  local_ep={local_ep}"
        f"  effective_ep={effective_ep}"
        f"  eval_every={eval_every}"
        f"  use_fedprox={_use_fedprox}"
        f"  device={device}"
        f"  AMP={use_amp}"
        f"  workers={NUM_WORKERS}"
        f"  sim={'on' if use_simulation else 'off'}"
    )

    # ---- Build initial model → serialise via state_dict ----
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
        _use_fedprox,
        device,
        use_amp=use_amp,
    )

    # ---- GPU fraction for Ray VCE ----
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
    print(f"\n  Simulation finished in {elapsed/60:.1f} min"
          f"  ({elapsed/num_rounds:.1f} s/round avg)")

    if strategy.last_params is None:
        raise RuntimeError(
            "No aggregated parameters captured. "
            "Check aggregate_fit() is returning non-None."
        )

    final_model = _load_final_model(cfg, strategy.last_params, device)
    torch.cuda.empty_cache()

    # ---- Per-client final evaluation (MC-Dropout) ----
    print("\n  Final per-client evaluation (MC-Dropout on trained global model):")
    results_row          = {"experiment": mode, "seed": seed}
    all_preds, all_trues = [], []

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
            "Quick sanity check (confirms FL trains without divergence):\n"
            "  --mode proposed --rounds 10 --local-epochs 1 --eval-every 2\n"
            "\n"
            "Full run for paper (fair vs E1 centralised, 50 epochs):\n"
            "  --mode proposed --rounds 50 --local-epochs 3 --eval-every 5\n"
            "\n"
            "Strongest anti-drift config (similarity-weighted + FedProx):\n"
            "  --mode proposed --rounds 50 --local-epochs 3 --use-fedprox\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode",
        choices=["fedavg", "fedprox", "proposed"],
        required=True,
    )
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--all_seeds", action="store_true")
    parser.add_argument(
        "--rounds",
        type=int, default=None,
        help="Override FL rounds. Recommended: 50.",
    )
    parser.add_argument(
        "--local-epochs",
        type=int, default=None, dest="local_epochs",
        help=(
            "Override epochs_local. "
            "IMPORTANT: use 1 if you observe RMSE increasing across rounds "
            "(client drift). Use 3 only once convergence is confirmed stable."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int, default=5, dest="eval_every",
        help="Evaluate every N rounds (default: 5 — saves ~80%% eval time).",
    )
    parser.add_argument(
        "--use-fedprox",
        action="store_true", dest="use_fedprox",
        help=(
            "Enable FedProx proximal term in 'proposed' mode. "
            "Adds μ/2·||w_local − w_global||² to constrain client drift. "
            "Recommended when local-epochs > 1. μ is set via federation.fedprox_mu "
            "in config.yaml (default 0.01)."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.rounds is not None:
        cfg["federation"]["num_rounds"] = args.rounds
    if args.local_epochs is not None:
        cfg["training"]["epochs_local"] = args.local_epochs
        rds = cfg["federation"]["num_rounds"]
        ep  = args.local_epochs
        print(f"[Override] epochs_local={ep}  "
              f"effective_training={rds * ep} steps total")

    if args.seed is not None:
        seeds = [args.seed]
    elif args.all_seeds:
        seeds = cfg["evaluation"]["seeds"]
    else:
        seeds = [cfg["reproducibility"]["seed"]]

    print(f"Experiment: {args.mode.upper()}  |  Seeds: {seeds}  "
          f"|  use_fedprox={args.use_fedprox}")
    for seed in seeds:
        run(
            cfg,
            args.mode,
            seed,
            eval_every  = args.eval_every,
            use_fedprox = args.use_fedprox,
        )

    print(f"\nDone. Results in {cfg['evaluation']['results_dir']}/{args.mode}.csv")


if __name__ == "__main__":
    main()