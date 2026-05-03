"""
federation/server.py
====================
Custom Flower server strategies.

Implements three aggregation strategies:
  1. FedAvg             — McMahan et al. (2017) baseline
  2. FedProx            — Li et al. (2020) baseline (proximal term in client)
  3. SimilarityWeighted — YOUR NOVELTY (Novelty 2)
                          Weights client updates by inverse KL-divergence
                          to the mean distribution. Clients whose data
                          distribution is closer to the global mean
                          receive higher aggregation weight.

All three produce parameters in the same format — swap strategies in
experiment scripts with one argument.
"""

import os
from functools import reduce
from logging import INFO
from typing import Optional

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


# Helper: weighted parameter aggregation

def _weighted_average(
    results: list[tuple[list[np.ndarray], int]],
    weights: np.ndarray,
) -> list[np.ndarray]:
    """
    Aggregate a list of (params, n_samples) using custom weights.
    weights: (n_clients,) non-negative, sum to 1.
    """
    aggregated = [
        sum(w * np.array(params[i]) for w, (params, _) in zip(weights, results))
        for i in range(len(results[0][0]))
    ]
    return aggregated


# Strategy 1: Standard FedAvg  (already in Flower — wrapped for consistency)

def build_fedavg(cfg: dict, on_fit_config_fn=None) -> FedAvg:
    """Standard FedAvg — McMahan et al. 2017."""
    return FedAvg(
        min_fit_clients         = cfg["federation"]["min_clients"],
        min_evaluate_clients    = cfg["federation"]["min_clients"],
        min_available_clients   = cfg["federation"]["min_clients"],
        on_fit_config_fn        = on_fit_config_fn,
        fit_metrics_aggregation_fn = _aggregate_fit_metrics,
    )


# Strategy 2 & 3: Custom strategies

class _BaseCustomStrategy(fl.server.strategy.Strategy):
    """
    Base class for custom FL strategies.
    Subclasses override aggregate_fit().
    """

    def __init__(self, cfg: dict, initial_params: list[np.ndarray]):
        self.cfg = cfg
        self.min_clients = cfg["federation"]["min_clients"]
        self._initial_params = ndarrays_to_parameters(initial_params)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self._initial_params

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
        losses = [loss * n for _, (loss, n, _) in
                  [(r, (r.loss, r.num_examples, r.metrics)) for _, r in results]]
        counts = [r.num_examples for _, r in results]
        avg_loss = sum(losses) / sum(counts)
        # Aggregate per-client metrics
        agg_metrics = _aggregate_eval_metrics(
            [(r.num_examples, r.metrics) for _, r in results]
        )
        return avg_loss, agg_metrics

    def evaluate(self, server_round, parameters):
        return None   # server-side eval not used; clients handle it

    def aggregate_fit(self, server_round, results, failures):
        raise NotImplementedError


class SimilarityWeightedStrategy(_BaseCustomStrategy):
    """
    Novelty 2: Similarity-Weighted Federated Aggregation.

    Client aggregation weight = exp(-KL(client_dist || mean_dist))
    Clients with data closer to the global mean receive more weight.

    This directly addresses the Non-IID problem: extreme outlier clients
    (FD002, FD004 with 6 operating conditions) are down-weighted slightly,
    preventing them from dominating the global model.

    Motivation for paper §3.4:
      Standard FedAvg weights by sample count, ignoring distribution shift.
      Our method additionally accounts for how representative each client's
      data is of the global distribution, measured by KL divergence.
    """

    def __init__(
        self,
        cfg:            dict,
        initial_params: list[np.ndarray],
        kl_matrix_path: str,
    ):
        super().__init__(cfg, initial_params)
        # Load pre-computed KL divergence matrix from preprocess.py output
        self.kl_matrix = pd.read_csv(kl_matrix_path, index_col=0)
        self.fd_keys   = cfg["data"]["clients"]
        self._compute_weights()

    def _compute_weights(self):
        """
        Compute static aggregation weights from KL divergence matrix.
        Weight_i = exp(-mean_KL_i) / sum_j(exp(-mean_KL_j))
        where mean_KL_i = average KL divergence from client i to all others.
        """
        mean_kl = np.array([
            self.kl_matrix.loc[fd, [f for f in self.fd_keys if f != fd]].mean()
            for fd in self.fd_keys
        ])
        # Softmax-style: lower KL → higher weight
        similarities = np.exp(-mean_kl)
        self.weights  = similarities / similarities.sum()

        print("\n[SimilarityWeighted] Aggregation weights:")
        for fd, w, kl in zip(self.fd_keys, self.weights, mean_kl):
            print(f"  {fd}: weight={w:.4f}  (mean_KL={kl:.4f})")

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        # Extract parameter arrays and sample counts
        weights_and_params = []
        for client_proxy, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            n      = fit_res.num_examples
            # Map client cid (str index) to aggregation weight
            cid    = int(client_proxy.cid)
            w      = self.weights[cid]
            weights_and_params.append((params, n, w))

        # Normalise weights (in case not all clients responded)
        ws     = np.array([w for _, _, w in weights_and_params])
        ws    /= ws.sum()
        agg    = [
            sum(w * np.array(p[i]) for (p, _, w), w in
                zip(weights_and_params, ws))
            for i in range(len(weights_and_params[0][0]))
        ]

        metrics = _aggregate_fit_metrics(
            [(n, fit_res.metrics) for _, fit_res in results
             if fit_res.metrics]
        )
        return ndarrays_to_parameters(agg), metrics


class FedProxStrategy(_BaseCustomStrategy):
    """
    FedProx strategy: same aggregation as FedAvg, but clients use proximal
    term in local loss. This strategy handles the server side (plain weighted
    average). The proximal term is applied client-side (see client.py).
    """

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        # Sample-count-weighted average (same as FedAvg)
        params_list = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total = sum(n for _, n in params_list)
        ws    = np.array([n / total for _, n in params_list])
        agg   = _weighted_average(params_list, ws)
        metrics = _aggregate_fit_metrics(
            [(n, fit_res.metrics) for _, fit_res in results if fit_res.metrics]
        )
        return ndarrays_to_parameters(agg), metrics


# Metric aggregation helpers

def _aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    keys  = metrics[0][1].keys()
    total = sum(n for n, _ in metrics)
    return {
        k: sum(n * m[k] for n, m in metrics) / total
        for k in keys
    }


def _aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
    if not metrics:
        return {}
    keys  = metrics[0][1].keys()
    total = sum(n for n, _ in metrics)
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
            "Choose: fedavg | fedprox | similarity_weighted"
        )