"""
federation/server.py
====================
Custom Flower server aggregation strategies.

Three strategies implemented:
  1. FedAvg             — McMahan et al. (2017) sample-count weighted avg
  2. FedProx            — Li et al. (2020) same aggregation as FedAvg;
                          proximal term is applied client-side (client.py)
  3. SimilarityWeighted — YOUR NOVELTY (Novelty 2)
                          Aggregation weight ∝ exp(−mean KL divergence)
                          Clients whose data distribution is closer to the
                          global mean receive higher aggregation weight,
                          directly addressing Non-IID degradation.
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

# Shared aggregation helper

def _weighted_average(
    results: list[tuple[list[np.ndarray], int]],
    weights: np.ndarray,
) -> list[np.ndarray]:
    """
    Weighted average of parameter arrays.

    Args:
        results: list of (param_arrays, n_samples) from each client
        weights: (n_clients,) normalised weights summing to 1
    """
    n_layers = len(results[0][0])
    return [
        sum(w * np.array(params[i])
            for w, (params, _) in zip(weights, results))
        for i in range(n_layers)
    ]


# Strategy 1: Standard FedAvg

def build_fedavg(cfg: dict) -> FedAvg:
    """Standard FedAvg — McMahan et al. 2017."""
    return FedAvg(
        min_fit_clients            = cfg["federation"]["min_clients"],
        min_evaluate_clients       = cfg["federation"]["min_clients"],
        min_available_clients      = cfg["federation"]["min_clients"],
        fit_metrics_aggregation_fn = _aggregate_fit_metrics,
    )


# Base class for custom strategies

class _BaseCustomStrategy(fl.server.strategy.Strategy):
    """Shared infrastructure for FedProx and SimilarityWeighted."""

    def __init__(self, cfg: dict, initial_params: list[np.ndarray]):
        self.cfg         = cfg
        self.min_clients = cfg["federation"]["min_clients"]
        self._initial    = ndarrays_to_parameters(initial_params)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self._initial

    def configure_fit(self, server_round, parameters, client_manager):
        config = {"round": server_round}
        sample = client_manager.sample(self.min_clients)
        return [(c, fl.common.FitIns(parameters, config)) for c in sample]

    def configure_evaluate(self, server_round, parameters, client_manager):
        config = {"round": server_round}
        sample = client_manager.sample(self.min_clients)
        return [(c, fl.common.EvaluateIns(parameters, config)) for c in sample]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total  = sum(r.num_examples for _, r in results)
        w_loss = sum(r.loss * r.num_examples for _, r in results) / total
        agg    = _aggregate_eval_metrics(
            [(r.num_examples, r.metrics) for _, r in results]
        )
        return w_loss, agg

    def evaluate(self, server_round, parameters):
        return None  # server-side eval delegated to clients

    def aggregate_fit(self, server_round, results, failures):
        raise NotImplementedError


# Strategy 2: FedProx (server side — proximal term is in client.py)

class FedProxStrategy(_BaseCustomStrategy):
    """
    Server-side FedProx: plain sample-count weighted average.
    The proximal regularisation term μ/2·||w−w̄||² is applied
    during local client training (see federation/client.py).
    """

    def aggregate_fit(self, server_round, results, failures):
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
            [(n, fit_res.metrics)
             for _, fit_res in results if fit_res.metrics is not None]
        )
        return ndarrays_to_parameters(agg), metrics


# Strategy 3: Similarity-Weighted Aggregation

class SimilarityWeightedStrategy(_BaseCustomStrategy):
    """
    Novelty 2: Distribution-Similarity-Weighted Federated Aggregation.

    Aggregation weight for client i:
      w_i = exp(−mean_KL_i) / Σ_j exp(−mean_KL_j)

    where mean_KL_i is the average KL divergence from client i's engine
    lifetime distribution to all other clients.

    Motivation (paper §3.4):
      Standard FedAvg weights by sample count, ignoring distribution shift.
      In a Non-IID federated PdM setting, clients with extreme distributions
      (e.g., FD002/FD004 with 6 operating conditions) would dominate if
      weighted by sample count alone, degrading the global model for other
      clients. Our method down-weights outlier clients proportionally to
      their KL divergence from the population mean.
    """

    def __init__(
        self,
        cfg:            dict,
        initial_params: list[np.ndarray],
        kl_matrix_path: str,
    ):
        super().__init__(cfg, initial_params)
        kl = pd.read_csv(kl_matrix_path, index_col=0)
        self.fd_keys = cfg["data"]["clients"]
        self._compute_weights(kl)

    def _compute_weights(self, kl_matrix: pd.DataFrame):
        """
        Compute static client weights from the pre-computed KL matrix.
        Clients are weighted once before training starts and held fixed.

        Stored as a dict keyed by fd string (e.g. "FD001") so that
        aggregate_fit() can look up weights via fit_res.metrics["fd"]
        instead of proxy.cid.  proxy.cid in Flower >= 1.x simulation is
        a large hash integer, NOT a sequential 0..N-1 index — indexing
        self.weights[int(proxy.cid)] was the direct cause of:
          IndexError: index 7310881596489040937 is out of bounds for axis 0
        """
        mean_kl = {
            fd: kl_matrix.loc[fd, [f for f in self.fd_keys if f != fd]].mean()
            for fd in self.fd_keys
        }
        similarities = {fd: np.exp(-mean_kl[fd]) for fd in self.fd_keys}
        total = sum(similarities.values())
        # weights_dict: fd → normalised aggregation weight (sums to 1)
        self.weights_dict = {fd: sim / total for fd, sim in similarities.items()}

        print("\n[SimilarityWeighted] Aggregation weights (from KL matrix):")
        for fd in self.fd_keys:
            print(f"  {fd}: weight={self.weights_dict[fd]:.4f}"
                  f"  (mean_KL={mean_kl[fd]:.4f})")

    def aggregate_fit(
        self,
        server_round: int,
        results:      list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        # Build (params, n_samples, orig_weight) triples.
        #
        # FIX: We look up weights via fit_res.metrics["fd"] — the fd string
        # that client.py now includes in every fit() return metrics dict.
        # This replaces the broken int(proxy.cid) index lookup: proxy.cid in
        # Flower >= 1.x simulation is a large hash (not 0..N-1), so
        # self.weights[int(proxy.cid)] raised IndexError.
        fallback_w = 1.0 / len(self.fd_keys)   # equal weight if fd missing
        client_data = []
        for proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            n      = fit_res.num_examples
            fd     = (fit_res.metrics or {}).get("fd", "")
            orig_w = self.weights_dict.get(fd, fallback_w)
            if not fd:
                print(f"  [WARN] SimilarityWeighted: no 'fd' in metrics for "
                      f"proxy.cid={proxy.cid}; using equal weight={fallback_w:.4f}")
            client_data.append((params, n, orig_w))

        # Renormalise in case fewer than min_clients responded
        raw_weights  = np.array([orig_w for _, _, orig_w in client_data])
        norm_weights = raw_weights / raw_weights.sum()

        # Weighted average
        n_layers = len(client_data[0][0])
        agg = [
            sum(
                norm_w * np.array(params[i])
                for (params, _n, _orig_w), norm_w
                in zip(client_data, norm_weights)
            )
            for i in range(n_layers)
        ]

        # Bug 4 fix: use "is not None" — empty dict {} is falsy but valid
        metrics = _aggregate_fit_metrics(
            [(n, fit_res.metrics)
             for _, fit_res in results if fit_res.metrics is not None]
        )
        return ndarrays_to_parameters(agg), metrics


# Metric aggregation helpers

def _aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    # Skip non-numeric fields (e.g. "fd" string tag added by clients so the
    # server can identify sub-datasets without relying on proxy.cid hashes).
    keys = [
        k for k in metrics[0][1].keys()
        if isinstance(metrics[0][1][k], (int, float))
    ]
    return {k: sum(n * m[k] for n, m in metrics) / total for k in keys}


def _aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys = [
        k for k in metrics[0][1].keys()
        if isinstance(metrics[0][1][k], (int, float))
    ]
    return {
        k: sum(n * m[k] for n, m in metrics if k in m) / total
        for k in keys
    }


# Factory

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