"""
experiments/ablation.py
========================
Experiment E5 — Ablation Study

Removes one component at a time from the full proposed system to isolate
the contribution of each novelty.

Variants (5 total):
  full          → Complete proposed system              (baseline for comparison)
  no_sim        → Remove Weibull augmentation           (ablates Novelty 1)
  no_sim_weight → Replace similarity weighting w/ FedAvg (ablates Novelty 2)
  no_phys_loss  → Replace hybrid loss with plain MSE   (ablates Novelty 3)
  no_attention  → Replace attention with global avg pool (ablates architecture)

Each variant is evaluated at final FL round using the actual trained global
model (loaded via _TrackingStrategy, same mechanism as train_federated.py).

Usage:
    python experiments/ablation.py                      # all variants, seed 42
    python experiments/ablation.py --all_seeds          # all variants × 5 seeds
    python experiments/ablation.py --variant no_sim     # single variant
    python experiments/ablation.py --variant no_sim --seed 123
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
from flwr.common import parameters_to_ndarrays

from evaluate import compute_metrics, make_loader, log_results
from federation.client import PDMClient
from federation.server import build_strategy
from models.loss import HybridRULLoss
from models.tcn import TCN, build_model


# Reproducibility
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Variant definitions
VARIANTS: dict[str, dict] = {
    "full": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  True,
        "description":    "Full proposed system",
    },
    "no_sim": {
        "use_simulation": False,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  True,
        "description":    "− Weibull simulation (Novelty 1)",
    },
    "no_sim_weight": {
        "use_simulation": True,
        "strategy":       "fedavg",
        "use_phys_loss":  True,
        "use_attention":  True,
        "description":    "− Similarity-weighted aggregation (Novelty 2)",
    },
    "no_phys_loss": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  False,
        "use_attention":  True,
        "description":    "− Physics monotonicity loss (Novelty 3)",
    },
    "no_attention": {
        "use_simulation": True,
        "strategy":       "similarity_weighted",
        "use_phys_loss":  True,
        "use_attention":  False,
        "description":    "− Temporal attention layer",
    },
}


# TCN without attention (ablation variant)
class _TCNNoAttention(nn.Module):
    """
    TCN with temporal attention replaced by global average pooling.
    Architecture is otherwise identical to the full model.
    """
    def __init__(self, base: TCN):
        super().__init__()
        self.tcn  = base.tcn
        self.head = base.head  # reuse same head — same parameter count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)        # (B, hidden, T)
        x = x.mean(dim=-1)     # global avg pool → (B, hidden)
        return self.head(x).squeeze(-1)

    def forward_with_hi(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_p  = x.permute(0, 2, 1)
        feat = self.tcn(x_p)                             # (B, hidden, T)
        hi   = torch.sigmoid(feat.mean(dim=1))           # (B, T)
        out  = self.head(feat.mean(dim=-1)).squeeze(-1)  # (B,)
        return out, hi


# MSE-only loss (replaces hybrid loss for no_phys_loss variant)
class _MSEOnlyLoss:
    """
    Drop-in replacement for HybridRULLoss that ignores the hi_seq argument.
    Returns the same (loss, components) tuple format so client.py is unchanged.
    """
    def __call__(
        self,
        rul_pred: torch.Tensor,
        rul_true: torch.Tensor,
        hi_seq:   torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss = nn.MSELoss()(rul_pred, rul_true)
        return loss, {"mse": loss.item(), "physics": 0.0, "total": loss.item()}


# Ablation client (extends PDMClient with variant flags)
class _AblationClient(PDMClient):
    """
    Wraps PDMClient to support ablation-specific overrides:
      - use_phys_loss=False → swap criterion for _MSEOnlyLoss
      - use_attention=False → swap model for _TCNNoAttention
    """

    def __init__(
        self,
        use_phys_loss: bool,
        use_attention: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Override criterion if physics loss is ablated
        if not use_phys_loss:
            self.criterion = _MSEOnlyLoss()

        # Override model if attention is ablated
        if not use_attention:
            self.model = _TCNNoAttention(self.model).to(self.device)

    def set_parameters(self, parameters: list[np.ndarray]):
        """Load parameters into whatever model variant is active."""
        state_dict = self.model.state_dict()
        if len(parameters) != len(state_dict):
            raise ValueError(
                f"Parameter count mismatch for {self.fd}: "
                f"model={len(state_dict)}, received={len(parameters)}"
            )
        new_state = {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in zip(state_dict.keys(), parameters)
        }
        self.model.load_state_dict(new_state, strict=True)

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [p.cpu().numpy() for p in self.model.parameters()]


# Tracking strategy (same decorator as train_federated.py)
class _TrackingStrategy(fl.server.strategy.Strategy):
    def __init__(self, wrapped: fl.server.strategy.Strategy):
        self._w = wrapped
        self.last_params: list[np.ndarray] | None = None

    def initialize_parameters(self, cm):
        return self._w.initialize_parameters(cm)

    def configure_fit(self, r, p, cm):
        return self._w.configure_fit(r, p, cm)

    def configure_evaluate(self, r, p, cm):
        return self._w.configure_evaluate(r, p, cm)

    def evaluate(self, r, p):
        return self._w.evaluate(r, p)

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = self._w.aggregate_fit(server_round, results, failures)
        if params is not None:
            self.last_params = parameters_to_ndarrays(params)
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        return self._w.aggregate_evaluate(server_round, results, failures)


# Load global model from captured FL parameters
def _load_model(
    cfg:           dict,
    params:        list[np.ndarray],
    use_attention: bool,
    device:        torch.device,
) -> nn.Module:
    base = build_model(cfg).to(device)
    model = base if use_attention else _TCNNoAttention(base).to(device)
    state_dict = model.state_dict()
    new_state = {
        k: torch.tensor(v, dtype=torch.float32).to(device)
        for k, v in zip(state_dict.keys(), params)
    }
    model.load_state_dict(new_state, strict=True)
    return model


# Run one ablation variant
def run_variant(cfg: dict, variant_name: str, seed: int) -> dict:
    v = VARIANTS[variant_name]
    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_simulation = v["use_simulation"]
    use_phys_loss  = v["use_phys_loss"]
    use_attention  = v["use_attention"]
    strategy_name  = v["strategy"]
    fd_keys        = cfg["data"]["clients"]

    print(f"\n[Ablation | {variant_name}]  seed={seed}  → {v['description']}")

    # Initial model (correct architecture for this variant)
    base_init = build_model(cfg)
    init_model = base_init if use_attention else _TCNNoAttention(base_init)
    init_params = [p.detach().numpy() for p in init_model.parameters()]

    base_strategy = build_strategy(strategy_name, cfg, init_params)
    strategy      = _TrackingStrategy(base_strategy)

    # Client factory
    def client_fn(cid: str) -> _AblationClient:
        fd = fd_keys[int(cid)]
        return _AblationClient(
            fd=fd,
            cfg=cfg,
            use_simulation=use_simulation,
            use_fedprox=False,         # FedProx not tested in ablation
            device=device,
            use_phys_loss=use_phys_loss,
            use_attention=use_attention,
        )

    # Run federation
    fl.simulation.start_simulation(
        client_fn        = client_fn,
        num_clients      = len(fd_keys),
        config           = fl.server.ServerConfig(
                               num_rounds=cfg["federation"]["num_rounds"]
                           ),
        strategy         = strategy,
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
    )

    if strategy.last_params is None:
        raise RuntimeError(f"No parameters captured for variant '{variant_name}'.")

    # Load actual trained global model
    final_model = _load_model(cfg, strategy.last_params, use_attention, device)

    # Evaluate per client
    results = {
        "experiment":  "ablation",
        "variant":     variant_name,
        "description": v["description"],
        "seed":        seed,
    }
    all_preds, all_trues = [], []

    for fd in fd_keys:
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
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

        print(f"  {fd}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  PHM={m['phm_score']:.1f}")
        results[f"{fd}_rmse"] = round(m["rmse"],      4)
        results[f"{fd}_mae"]  = round(m["mae"],       4)
        results[f"{fd}_phm"]  = round(m["phm_score"], 2)

    overall = compute_metrics(
        np.concatenate(all_preds), np.concatenate(all_trues)
    )
    results["overall_rmse"] = round(overall["rmse"],      4)
    results["overall_mae"]  = round(overall["mae"],       4)
    results["overall_phm"]  = round(overall["phm_score"], 2)
    print(f"  OVERALL: RMSE={overall['rmse']:.2f}  "
          f"MAE={overall['mae']:.2f}  PHM={overall['phm_score']:.1f}")

    return results


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="Ablation study (E5) — removes one component at a time."
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--variant",   choices=list(VARIANTS.keys()), default=None,
                        help="Single variant. Omit to run all 5.")
    parser.add_argument("--seed",      type=int, default=None)
    parser.add_argument("--all_seeds", action="store_true",
                        help="Run all seeds from config for each variant.")
    parser.add_argument("--rounds",    type=int, default=None,
                        help="Override FL rounds (e.g. --rounds 10 for quick test).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.rounds:
        cfg["federation"]["num_rounds"] = args.rounds

    variants = [args.variant] if args.variant else list(VARIANTS.keys())
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = cfg["evaluation"]["seeds"] if args.all_seeds else [
            cfg["reproducibility"]["seed"]
        ]

    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ablation.csv")

    total_runs = len(variants) * len(seeds)
    print(f"Ablation Study — {len(variants)} variant(s) × {len(seeds)} seed(s) = {total_runs} runs")

    for variant in variants:
        for seed in seeds:
            row = run_variant(cfg, variant, seed)
            log_results(row, csv_path)

    print(f"\nAll results saved to {csv_path}")

    # Print summary table (seed 42 only)
    first_seed = str(cfg["reproducibility"]["seed"])
    print(f"\n{'='*72}")
    print(f"  ABLATION SUMMARY (seed={first_seed})")
    print(f"{'='*72}")
    print(f"  {'Variant':<22} {'Description':<38} {'RMSE':>7} {'MAE':>7}")
    print(f"  {'-'*70}")
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("seed") == first_seed:
                    print(f"  {row['variant']:<22} {row['description']:<38} "
                          f"{float(row['overall_rmse']):>7.2f} "
                          f"{float(row['overall_mae']):>7.2f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()