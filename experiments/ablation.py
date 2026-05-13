"""
experiments/ablation.py
========================
Experiment E5 — Ablation Study

Removes one component at a time from the full proposed system to isolate
the contribution of each novelty.

Variants (6 total):
  full          → Complete proposed system                     (baseline)
  no_sim        → Remove Weibull augmentation                  (ablates Novelty 1)
  no_sim_weight → Replace similarity weighting with FedAvg     (ablates Novelty 2)
  no_phys_loss  → Replace hybrid loss with plain MSE           (ablates Novelty 3)
  no_attention  → Replace attention with global avg pool       (ablates architecture)
  no_fedprox    → Remove proximal drift constraint (FedProx)   (ablates FedProx)
"""

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import yaml
import flwr as fl
from flwr.common import Context, parameters_to_ndarrays

from evaluate import compute_metrics, make_loader, log_results
from federation.client import PDMClient
from federation.server import build_strategy
from models.loss import HybridRULLoss
from models.tcn import TCN, build_model


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Every variant now carries use_fedprox explicitly so client_fn can read it
# directly instead of relying on a hardcoded default.
#
# The "full" variant uses use_fedprox=True because that is what produced the
# 18.24 RMSE reported in Table 3.  Any variant that is NOT testing FedProx
# also uses use_fedprox=True so we isolate one variable at a time — the gold
# standard for ablation study design.

VARIANTS: dict[str, dict] = {
    "full": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  True,
        "use_fedprox":    True,
        "description":    "Full proposed system",
    },
    "no_sim": {
        "use_simulation": False,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  True,
        "use_fedprox":    True,
        "description":    "− Weibull simulation (Novelty 1)",
    },
    "no_sim_weight": {
        "use_simulation": True,
        "strategy":       "fedavg",
        "use_phys_loss":  True,
        "use_attention":  True,
        "use_fedprox":    True,
        "description":    "− Similarity-weighted aggregation (Novelty 2)",
    },
    "no_phys_loss": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  False,
        "use_attention":  True,
        "use_fedprox":    True,
        "description":    "− Physics monotonicity loss (Novelty 3)",
    },
    "no_attention": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  False,
        "use_fedprox":    True,
        "description":    "− Temporal attention layer",
    },
    "no_fedprox": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  True,
        "use_fedprox":    False,
        "description":    "− Proximal drift constraint (FedProx)",
    },
}


# ---------------------------------------------------------------------------
# Architecture variant: TCN without attention
# ---------------------------------------------------------------------------

class _TCNNoAttention(nn.Module):
    """
    TCN with temporal attention replaced by global average pooling.
    Architecture is otherwise identical to the full model.
    """
    def __init__(self, base: TCN):
        super().__init__()
        self.tcn  = base.tcn
        self.head = base.head   # reuse same head — keeps parameter count equal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)         # (B, hidden, T)
        x = x.mean(dim=-1)      # global avg pool → (B, hidden)
        return self.head(x).squeeze(-1)

    def forward_with_hi(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_p  = x.permute(0, 2, 1)
        feat = self.tcn(x_p)                             # (B, hidden, T)
        hi   = torch.sigmoid(feat.mean(dim=1))           # (B, T)
        out  = self.head(feat.mean(dim=-1)).squeeze(-1)  # (B,)
        return out, hi


# ---------------------------------------------------------------------------
# Loss variant: MSE-only (replaces hybrid loss for no_phys_loss)
# ---------------------------------------------------------------------------

class _MSEOnlyLoss:
    """
    Drop-in replacement for HybridRULLoss.
    Returns the same (loss, components) tuple so client.py is unchanged.
    """
    def __call__(
        self,
        rul_pred: torch.Tensor,
        rul_true: torch.Tensor,
        hi_seq:   torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss = nn.MSELoss()(rul_pred, rul_true)
        return loss, {"mse": loss.item(), "physics": 0.0, "total": loss.item()}


# ---------------------------------------------------------------------------
# Ablation client
# ---------------------------------------------------------------------------

class _AblationClient(PDMClient):
    """
    Extends PDMClient with variant-specific overrides.
      use_phys_loss=False  → swap criterion for _MSEOnlyLoss
      use_attention=False  → swap model for _TCNNoAttention
    """

    def __init__(
        self,
        use_phys_loss: bool,
        use_attention: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not use_phys_loss:
            self.criterion = _MSEOnlyLoss()

        if not use_attention:
            self.model = _TCNNoAttention(self.model).to(self.device)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load parameters into whatever model variant is currently active."""
        state_dict = self.model.state_dict()
        if len(parameters) != len(state_dict):
            raise ValueError(
                f"Parameter count mismatch for {self.fd}: "
                f"model has {len(state_dict)} tensors, "
                f"received {len(parameters)}."
            )
        new_state = {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in zip(state_dict.keys(), parameters)
        }
        self.model.load_state_dict(new_state, strict=True)

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        # FIX: state_dict().values() detached natively, ensuring length perfectly matches
        return [val.cpu().numpy() for val in self.model.state_dict().values()]


# ---------------------------------------------------------------------------
# Tracking strategy
# ---------------------------------------------------------------------------

class _TrackingStrategy(fl.server.strategy.Strategy):
    """
    Decorator around any Flower strategy that:
      (a) captures the last aggregated parameters for post-hoc evaluation, and
      (b) skips client-side evaluation on rounds not divisible by eval_every.
    """

    def __init__(
        self,
        wrapped:    fl.server.strategy.Strategy,
        eval_every: int = 5,
    ):
        self._w         = wrapped
        self.eval_every = eval_every
        self.last_params: list[np.ndarray] | None = None

    def initialize_parameters(self, client_manager=None, **kwargs):
        return self._w.initialize_parameters(client_manager=client_manager, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        return self._w.configure_fit(server_round, parameters, client_manager, **kwargs)

    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
        # Skip evaluation on non-designated rounds to cut wall-clock time.
        if server_round % self.eval_every != 0:
            return []
        return self._w.configure_evaluate(server_round, parameters, client_manager, **kwargs)

    def evaluate(self, server_round, parameters, **kwargs):
        return self._w.evaluate(server_round, parameters, **kwargs)

    def aggregate_fit(self, server_round, results, failures, **kwargs):
        params, metrics = self._w.aggregate_fit(server_round, results, failures, **kwargs)
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures, **kwargs):
        return self._w.aggregate_evaluate(server_round, results, failures, **kwargs)


# ---------------------------------------------------------------------------
# Helper: load final global model from captured FL parameters
# ---------------------------------------------------------------------------

def _load_model(
    cfg:           dict,
    params:        list[np.ndarray],
    use_attention: bool,
    device:        torch.device,
) -> nn.Module:
    base  = build_model(cfg).to(device)
    model = base if use_attention else _TCNNoAttention(base).to(device)
    state_dict = model.state_dict()
    new_state = {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in zip(state_dict.keys(), params)
    }
    model.load_state_dict(new_state, strict=True)
    return model


# ---------------------------------------------------------------------------
# Run one variant
# ---------------------------------------------------------------------------

def run_variant(
    cfg:          dict,
    variant_name: str,
    seed:         int,
    eval_every:   int = 5,
) -> dict:
    """
    Execute a full FL simulation for one ablation variant, then evaluate the
    final global model on every client's held-out test set.

    Returns a flat dict suitable for CSV logging via log_results().
    """
    v = VARIANTS[variant_name]
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read variant flags
    use_simulation = v["use_simulation"]
    use_phys_loss  = v["use_phys_loss"]
    use_attention  = v["use_attention"]
    use_fedprox    = v["use_fedprox"]
    strategy_name  = v["strategy"]
    fd_keys        = cfg["data"]["clients"]

    print(
        f"\n[Ablation | {variant_name}]  seed={seed}  "
        f"fedprox={'on' if use_fedprox else 'off'}  "
        f"eval_every={eval_every}  → {v['description']}"
    )

    # Build initial model (correct architecture for this variant)
    base_init  = build_model(cfg)
    init_model = base_init if use_attention else _TCNNoAttention(base_init)
    
    # FIX: Extract state_dict().values() to perfectly match what clients expect
    init_params = [val.cpu().numpy() for val in init_model.state_dict().values()]

    base_strategy = build_strategy(strategy_name, cfg, init_params)
    strategy      = _TrackingStrategy(base_strategy, eval_every=eval_every)

    # Client factory - FIX: Uses 'Context' instead of 'cid: str'
    def client_fn(context: Context):
        partition_id = context.node_config.get(
            "partition-id", int(context.node_id)
        )
        fd = fd_keys[int(partition_id) % len(fd_keys)]
        
        # FIX: Returns .to_client() directly to satisfy Flower's updated API
        return _AblationClient(
            fd             = fd,
            cfg            = cfg,
            use_simulation = use_simulation,
            use_fedprox    = use_fedprox,
            device         = device,
            use_phys_loss  = use_phys_loss,
            use_attention  = use_attention,
        ).to_client()

    # Calculate safe GPU fraction to prevent CUDA OOM / Race conditions
    gpu_fraction = 1.0 / len(fd_keys) if torch.cuda.is_available() else 0.0

    # Run FL simulation
    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = len(fd_keys),
        config           = fl.server.ServerConfig(
                               num_rounds=cfg["federation"]["num_rounds"]
                           ),
        strategy         = strategy,
        client_resources = {"num_cpus": 2, "num_gpus": gpu_fraction},
    )

    if strategy.last_params is None:
        raise RuntimeError(
            f"No parameters were captured for variant '{variant_name}'. "
            "Check that aggregate_fit ran at least once."
        )

    # Load the actual final trained global model
    final_model = _load_model(cfg, strategy.last_params, use_attention, device)

    # Per-client evaluation
    results = {
        "experiment":  "ablation",
        "variant":     variant_name,
        "description": v["description"],
        "seed":        seed,
    }
    all_preds, all_trues = [], []

    for fd in fd_keys:
        npz    = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        loader = make_loader(
            npz["X_test"], npz["y_test"], cfg["training"]["batch_size"]
        )
        final_model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                preds.append(final_model(X.to(device)).cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        m     = compute_metrics(preds, trues)
        all_preds.append(preds)
        all_trues.append(trues)

        print(
            f"  {fd}: RMSE={m['rmse']:.2f}  "
            f"MAE={m['mae']:.2f}  PHM={m['phm_score']:.1f}"
        )
        results[f"{fd}_rmse"] = round(m["rmse"],      4)
        results[f"{fd}_mae"]  = round(m["mae"],       4)
        results[f"{fd}_phm"]  = round(m["phm_score"], 2)

    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results["overall_rmse"] = round(overall["rmse"],      4)
    results["overall_mae"]  = round(overall["mae"],       4)
    results["overall_phm"]  = round(overall["phm_score"], 2)
    print(
        f"  OVERALL: RMSE={overall['rmse']:.2f}  "
        f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}"
    )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ablation study (E5) — removes one component at a time.\n"
            "\n"
            "Recommended commands:\n"
            "  Sanity check (1 seed, fast):\n"
            "    python experiments/ablation.py "
            "--seed 42 --rounds 100 --local-epochs 1 --eval-every 5\n"
            "\n"
            "  Full Q1 run (5 seeds):\n"
            "    python experiments/ablation.py "
            "--all_seeds --local-epochs 1 --eval-every 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",     default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--variant",    choices=list(VARIANTS.keys()), default=None,
        help="Single variant to run. Omit to run all 6.",
    )
    parser.add_argument(
        "--seed",       type=int, default=None,
        help="Single seed. Omit to use config default or --all_seeds.",
    )
    parser.add_argument(
        "--all_seeds",  action="store_true",
        help="Run all seeds from config.evaluation.seeds for each variant.",
    )
    parser.add_argument(
        "--rounds",     type=int, default=100,
        help="Override federation.num_rounds (Default: 100 to allow noise phase-out).",
    )
    parser.add_argument(
        "--local-epochs", dest="local_epochs", type=int, default=None,
        help=(
            "Override training.epochs_local. "
            "Use --local-epochs 1 for faster sanity checks."
        ),
    )
    parser.add_argument(
        "--eval-every", dest="eval_every", type=int, default=None,
        help=(
            "Evaluate clients every N rounds (default: federation.eval_every "
            "from config, or 5 if not set). "
            "Crucially prevents the ~3-day runtime of evaluating every round."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides to config in-place
    if args.rounds:
        cfg["federation"]["num_rounds"] = args.rounds
    if args.local_epochs:
        cfg["training"]["epochs_local"] = args.local_epochs
    if args.eval_every:
        cfg["federation"]["eval_every"] = args.eval_every

    # Resolve eval_every: CLI > config > default of 5
    eval_every = cfg["federation"].get("eval_every", 5)

    # Resolve variants and seeds
    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    if args.seed is not None:
        seeds = [args.seed]
    elif args.all_seeds:
        seeds = cfg["evaluation"]["seeds"]
    else:
        seeds = [cfg["reproducibility"]["seed"]]

    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ablation.csv")

    total_runs = len(variants) * len(seeds)
    print(
        f"\nAblation Study (E5)\n"
        f"  Variants   : {len(variants)} ({', '.join(variants)})\n"
        f"  Seeds      : {len(seeds)} ({seeds})\n"
        f"  Total runs : {total_runs}\n"
        f"  FL rounds  : {cfg['federation']['num_rounds']}\n"
        f"  Local epochs: {cfg['training']['epochs_local']}\n"
        f"  Eval every : every {eval_every} rounds\n"
        f"  CSV output : {csv_path}\n"
    )

    for variant in variants:
        for seed in seeds:
            row = run_variant(cfg, variant, seed, eval_every=eval_every)
            log_results(row, csv_path)

    print(f"\nAll results saved → {csv_path}")

    # Summary table (first seed only)
    first_seed = str(seeds[0])
    print(f"\n{'='*80}")
    print(f"  ABLATION SUMMARY  (seed={first_seed})")
    print(f"{'='*80}")
    print(f"  {'Variant':<22} {'FedProx':>8} {'Description':<42} "
          f"{'RMSE':>7} {'MAE':>7} {'PHM':>8}")
    print(f"  {'-'*78}")

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("seed") == first_seed:
                    vname = row["variant"]
                    fp    = "on" if VARIANTS[vname]["use_fedprox"] else "off"
                    print(
                        f"  {vname:<22} {fp:>8} "
                        f"{row['description']:<42} "
                        f"{float(row['overall_rmse']):>7.2f} "
                        f"{float(row['overall_mae']):>7.2f} "
                        f"{float(row['overall_phm']):>8.1f}"
                    )
    except Exception:
        pass

    print(f"\n  The 'full' row MUST match the RMSE reported in Table 3.")
    print(f"  If it does not, check federation.fedprox_mu and strategy config.")


if __name__ == "__main__":
    main()