"""
federation/server.py
====================
Custom Flower server aggregation strategies.

Three strategies implemented:
  1. FedAvg             — McMahan et al. (2017) sample-count weighted avg
  2. FedProx            — Li et al. (2020) same aggregation as FedAvg;
                          proximal term is applied client-side (client.py)
  3. SimilarityWeighted — Novelty 2 (v2, fixed)

FIXES IN THIS VERSION
=====================
Fix A — Softer KL penalty  (kl_alpha defaults to 0.3, was implicitly 1.0)
  Original: w_i ∝ exp(−KL_i)
  Fixed:    w_i ∝ exp(−α · KL_i)

  With α=1.0, FD004's mean_KL=2.01 gives exp(−2.01)=0.134 → after
  normalisation ≈ 9.6 % weight — despite FD004 having 54 k training
  samples and 34 k test samples (≈48 % of total test data).

  With α=0.3: exp(−0.60)=0.549 → ≈ 22 % pre-blend similarity weight.
  Much less aggressive; distribution differences are still respected.

Fix B — Hybrid weight: blend similarity with sample-count fraction
  blended_i = λ · sim_i + (1−λ) · (n_i / Σn)
  Default λ=0.5. This prevents a large-data client from being ignored
  purely because its distribution diverges (FD002/FD004 have 6 operating
  conditions, naturally higher KL, but contain critical training signal).

Fix C — Weight floor
  Each client's similarity weight is floored at weight_floor/n_clients
  before blending. With floor=0.10 and 4 clients, no client's similarity
  component drops below 2.5 %. Applied before blending so the floor is
  in the similarity space, not the final blended space.

Fix D — Server-side learning rate (aggregation dampening)
  After computing the weighted average w_agg, apply:
    w_global_new = w_global_prev + server_lr · (w_agg − w_global_prev)
  server_lr=1.0 → standard FedAvg (identical behaviour)
  server_lr=0.8 → 20 % dampening, smooths oscillations from client drift.

  Motivation: with local_epochs > 1, clients drift toward their local
  optima. After aggregation the global model "bounces" between diverged
  client models. server_lr < 1.0 damps this bounce; each round the
  global model moves only 80 % of the way to the aggregate.

ALL FOUR PARAMETERS are configurable in config.yaml under federation:
  kl_alpha:     0.3
  weight_floor: 0.10
  blend_lambda: 0.5
  server_lr:    0.8
"""

import os
from typing import Optional

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# ------------------------------------------------------------------ #
# Shared aggregation helper
# ------------------------------------------------------------------ #

def _weighted_average(
    results: list[tuple[list[np.ndarray], int]],
    weights: np.ndarray,
) -> list[np.ndarray]:
    n_layers = len(results[0][0])
    return [
        sum(w * np.array(params[i])
            for w, (params, _) in zip(weights, results))
        for i in range(n_layers)
    ]


# ------------------------------------------------------------------ #
# Strategy 1: Standard FedAvg
# ------------------------------------------------------------------ #

def build_fedavg(cfg: dict) -> FedAvg:
    """
    Standard FedAvg — McMahan et al. 2017.
    Passes round + num_rounds to clients (enables inter-round LR decay).
    """
    num_rounds = cfg["federation"]["num_rounds"]

    def fit_config(server_round: int) -> dict:
        return {"round": server_round, "num_rounds": num_rounds}

    return FedAvg(
        min_fit_clients            = cfg["federation"]["min_clients"],
        min_evaluate_clients       = cfg["federation"]["min_clients"],
        min_available_clients      = cfg["federation"]["min_clients"],
        on_fit_config_fn           = fit_config,
        fit_metrics_aggregation_fn = _aggregate_fit_metrics,
    )


# ------------------------------------------------------------------ #
# Base class for custom strategies
# ------------------------------------------------------------------ #

class _BaseCustomStrategy(fl.server.strategy.Strategy):
    """Shared infrastructure for FedProx and SimilarityWeighted."""

    def __init__(self, cfg: dict, initial_params: list[np.ndarray]):
        self.cfg         = cfg
        self.min_clients = cfg["federation"]["min_clients"]
        self._initial    = ndarrays_to_parameters(initial_params)

    def initialize_parameters(self, client_manager=None, **kwargs) -> Optional[Parameters]:
        return self._initial

    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        config = {
            "round":      server_round,
            "num_rounds": self.cfg["federation"]["num_rounds"],
        }
        sample = client_manager.sample(self.min_clients)
        return [(c, fl.common.FitIns(parameters, config)) for c in sample]

    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
        config = {"round": server_round}
        sample = client_manager.sample(self.min_clients)
        return [(c, fl.common.EvaluateIns(parameters, config)) for c in sample]

    def aggregate_evaluate(self, server_round, results, failures, **kwargs):
        if not results:
            return None, {}
        total  = sum(r.num_examples for _, r in results)
        w_loss = sum(r.loss * r.num_examples for _, r in results) / total
        agg    = _aggregate_eval_metrics(
            [(r.num_examples, r.metrics) for _, r in results]
        )
        return w_loss, agg

    def evaluate(self, server_round, parameters, **kwargs):
        return None

    def aggregate_fit(self, server_round, results, failures, **kwargs):
        raise NotImplementedError


# ------------------------------------------------------------------ #
# Strategy 2: FedProx
# ------------------------------------------------------------------ #

class FedProxStrategy(_BaseCustomStrategy):
    """
    Server-side FedProx: plain sample-count weighted average.
    The proximal regularisation term μ/2·||w−w̄||² is applied
    during local client training (see federation/client.py).
    """

    def aggregate_fit(self, server_round, results, failures, **kwargs):
        if not results:
            return None, {}
        params_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total = sum(n for _, n in params_list)
        ws    = np.array([n / total for _, n in params_list])
        agg   = _weighted_average(params_list, ws)
        metrics = _aggregate_fit_metrics(
            [(fit_res.num_examples, fit_res.metrics)
             for _, fit_res in results if fit_res.metrics is not None]
        )
        return ndarrays_to_parameters(agg), metrics


# ------------------------------------------------------------------ #
# Strategy 3: Similarity-Weighted Aggregation (v2)
# ------------------------------------------------------------------ #

class SimilarityWeightedStrategy(_BaseCustomStrategy):
    """
    Novelty 2: Distribution-Similarity-Weighted Federated Aggregation (v2).

    Final weight for client i (four-step pipeline):
      1. sim_i    = exp(−α · mean_KL_i)             [Fix A: softer α]
      2. floored  = max(sim_i / Σsim, floor/n)       [Fix C: floor]
      3. renorm   → floored / Σfloored
      4. blended  = λ · floored_norm + (1−λ) · (n_i/Σn)  [Fix B: hybrid]
      5. renorm   → blended / Σblended

    Global update:
      w_global ← w_prev + server_lr · (w_agg − w_prev)   [Fix D: damping]

    All four hyperparameters live under federation: in config.yaml.
    """

    def __init__(
        self,
        cfg:            dict,
        initial_params: list[np.ndarray],
        kl_matrix_path: str,
    ):
        super().__init__(cfg, initial_params)
        fed_cfg      = cfg.get("federation", {})
        self.fd_keys = cfg["data"]["clients"]

        # ---- Hyperparameters ----
        # Fix A: softer KL exponent.  Lower → less aggressive down-weighting.
        self.kl_alpha = float(fed_cfg.get("kl_alpha", 0.3))

        # Fix C: minimum weight fraction per client (applied in similarity space).
        # With 4 clients and floor=0.10, each client's sim weight ≥ 0.025.
        self.weight_floor = float(fed_cfg.get("weight_floor", 0.10))

        # Fix B: blend ratio.  0 = pure sample-count, 1 = pure similarity.
        self.blend_lambda = float(fed_cfg.get("blend_lambda", 0.5))

        # Fix D: server-side LR dampening.  1.0 = standard FedAvg.
        self.server_lr = float(fed_cfg.get("server_lr", 0.8))

        # Storage for Fix D momentum
        self._prev_params: list[np.ndarray] | None = None

        kl = pd.read_csv(kl_matrix_path, index_col=0)
        self._compute_sim_weights(kl)

    def _compute_sim_weights(self, kl_matrix: pd.DataFrame):
        """
        Compute and store static per-client similarity weights.
        These are the *pre-blend* weights; sample-count blending happens
        in aggregate_fit() when n_i is available.
        """
        mean_kl = {
            fd: kl_matrix.loc[fd, [f for f in self.fd_keys if f != fd]].mean()
            for fd in self.fd_keys
        }

        # Fix A: softer exponential
        raw   = {fd: np.exp(-self.kl_alpha * mean_kl[fd]) for fd in self.fd_keys}
        total = sum(raw.values())
        sim_w = {fd: v / total for fd, v in raw.items()}

        # Fix C: floor each client's similarity weight
        n         = len(self.fd_keys)
        floor_val = self.weight_floor / n          # per-client absolute floor
        floored   = {fd: max(sim_w[fd], floor_val) for fd in self.fd_keys}
        total_f   = sum(floored.values())
        # sim_weights_dict: floored, renormalised → pure similarity component
        self.sim_weights_dict = {fd: v / total_f for fd, v in floored.items()}

        print("\n[SimilarityWeighted v2] Pre-blend similarity weights:")
        print(f"  α={self.kl_alpha}  floor={self.weight_floor}"
              f"  λ={self.blend_lambda}  server_lr={self.server_lr}")
        for fd in self.fd_keys:
            # Show both old (α=1) and new weights for comparison
            old_raw  = np.exp(-1.0 * mean_kl[fd])
            print(f"  {fd}: new_sim={self.sim_weights_dict[fd]:.4f}"
                  f"  mean_KL={mean_kl[fd]:.4f}"
                  f"  (α=1.0 unnorm={old_raw:.4f})")

    def aggregate_fit(
        self,
        server_round: int,
        results:      list[tuple[ClientProxy, FitRes]],
        failures,
        **kwargs
    ) -> tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        # ---- Build per-client data ----
        fallback_w = 1.0 / len(self.fd_keys)
        client_data: list[tuple[list[np.ndarray], int, float, str]] = []
        for proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            n      = fit_res.num_examples
            fd     = (fit_res.metrics or {}).get("fd", "")
            sim_w  = self.sim_weights_dict.get(fd, fallback_w)
            if not fd:
                print(f"  [WARN] no 'fd' in metrics for proxy.cid={proxy.cid}; "
                      f"using equal weight={fallback_w:.4f}")
            client_data.append((params, n, sim_w, fd))

        # ---- Fix B: Hybrid weight = λ·sim + (1−λ)·sample_fraction ----
        total_n = sum(n for _, n, _, _ in client_data)
        blended: list[float] = []
        for _, n, sim_w, _ in client_data:
            sample_frac = n / total_n
            blended.append(
                self.blend_lambda * sim_w + (1.0 - self.blend_lambda) * sample_frac
            )

        # Renormalise
        total_b      = sum(blended)
        norm_weights = [w / total_b for w in blended]

        # Diagnostics: print every 5 rounds
        if server_round == 1 or server_round % 5 == 0:
            print(f"\n  [SW-Agg] Round {server_round} — final blended weights:")
            for (_, n, sim_w, fd), norm_w in zip(client_data, norm_weights):
                print(f"    {fd}: sim={sim_w:.3f}"
                      f"  sample_frac={n/total_n:.3f}"
                      f"  blended={norm_w:.3f}")

        # ---- Weighted average ----
        n_layers = len(client_data[0][0])
        agg = [
            sum(
                norm_w * np.array(params[i])
                for (params, _, _, _), norm_w in zip(client_data, norm_weights)
            )
            for i in range(n_layers)
        ]

        # ---- Fix D: server-side learning rate (aggregation dampening) ----
        # w_new = w_prev + server_lr * (w_agg - w_prev)
        # server_lr=1.0 → identical to standard FedAvg
        if self._prev_params is not None and self.server_lr < 1.0:
            agg = [
                prev + self.server_lr * (new - prev)
                for prev, new in zip(self._prev_params, agg)
            ]

        # Store copy for next round's dampening step
        self._prev_params = [
            a.copy() if hasattr(a, "copy") else np.array(a)
            for a in agg
        ]

        metrics = _aggregate_fit_metrics(
            [(n, fit_res.metrics)
             for _, fit_res in results if fit_res.metrics is not None]
        )
        return ndarrays_to_parameters(agg), metrics


# ------------------------------------------------------------------ #
# Metric aggregation helpers
# ------------------------------------------------------------------ #

def _aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys  = [
        k for k in metrics[0][1].keys()
        if isinstance(metrics[0][1][k], (int, float))
    ]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


def _aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys  = [
        k for k in metrics[0][1].keys()
        if isinstance(metrics[0][1][k], (int, float))
    ]
    return {
        k: sum(n * m[k] for n, m in metrics if k in m) / total
        for k in keys
    }


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def build_strategy(
    strategy_name:  str,
    cfg:            dict,
    initial_params: list[np.ndarray],
) -> fl.server.strategy.Strategy:
    """
    Args:
        strategy_name: "fedavg" | "fedprox" | "similarity_weighted"
    """
    name = strategy_name.lower()

    if name == "fedavg":
        return build_fedavg(cfg)

    elif name == "fedprox":
        return FedProxStrategy(cfg, initial_params)

    elif name == "similarity_weighted":
        kl_path = os.path.join(
            cfg["data"]["output_dir"], "kl_divergence_matrix.csv"
        )
        return SimilarityWeightedStrategy(cfg, initial_params, kl_path)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            "Options: fedavg | fedprox | similarity_weighted"
        )