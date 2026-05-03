"""
simulation/weibull.py
=====================
Physics-based synthetic data generator using calibrated Weibull degradation.

This is Novelty 1 of the paper:
  "Physics-constrained Weibull simulation injected locally per FL client,
   targeting the sparse late-degradation region (RUL <= 50)."

Usage:
    from simulation.weibull import WeibullSimulator
    sim = WeibullSimulator(k=2.3, lam=180.0, cfg=cfg)
    X_sim, y_sim = sim.generate(n_trajectories=300)
"""

import json
import os

import numpy as np
from scipy.stats import weibull_min


class WeibullSimulator:
    """
    Generates synthetic sensor degradation trajectories using the Weibull CDF
    as the underlying degradation model.

    Design choices (each must be justified in paper §3):
    - Weibull CDF: models cumulative failure probability — well-established for
      rotating machinery (Jardine et al., 2006)
    - Per-sensor sensitivity weights: different sensors respond at different rates
    - Late-degradation focus: only windows with RUL <= threshold are kept
      because individual FL clients always lack these samples
    - Additive Gaussian noise: calibrated to match real CMAPSS sensor SNR
    """

    def __init__(self, k: float, lam: float, cfg: dict, seed: int = 42):
        self.k   = k          # Weibull shape  (k > 1 → wear-out failure mode)
        self.lam = lam        # Weibull scale  (related to characteristic life)
        self.seed = seed

        sim_cfg          = cfg["simulation"]
        self.n_sensors   = len(cfg["data"]["selected_sensors"])
        self.window_size = cfg["data"]["window_size"]
        self.max_rul     = cfg["data"]["max_rul"]
        self.noise_std   = sim_cfg["noise_std"]
        self.max_lifetime= sim_cfg["max_lifetime"]
        self.threshold   = sim_cfg["late_rul_threshold"]

    def _single_trajectory(
        self, seed_offset: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate one run-to-failure trajectory.
        Returns:
            signals: (T, N_sensors) float32
            rul:     (T,)           float32
        """
        rng = np.random.default_rng(self.seed + seed_offset)

        # Sample lifetime from calibrated Weibull
        T = int(weibull_min.rvs(self.k, scale=self.lam,
                                random_state=self.seed + seed_offset))
        T = max(T, self.window_size + 5)
        T = min(T, self.max_lifetime)

        cycles = np.linspace(0, T, T)

        # Weibull CDF as degradation index — monotonically 0→1
        deg = weibull_min.cdf(cycles, c=self.k, scale=self.lam)  # (T,)

        # Per-sensor sensitivity — each sensor degrades at a different rate
        weights = rng.uniform(0.3, 1.0, self.n_sensors)           # (N,)
        signals = np.outer(deg, weights)                           # (T, N)

        # Additive noise calibrated to ~20 dB SNR (matching CMAPSS noise floor)
        signals += rng.normal(0, self.noise_std, signals.shape)
        signals  = np.clip(signals, 0.0, 1.0).astype(np.float32)

        rul = np.maximum(0, np.minimum(self.max_rul, T - cycles)).astype(np.float32)
        return signals, rul

    def generate(
        self, n_trajectories: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate n_trajectories synthetic sequences.
        Only windows with RUL <= self.threshold are kept
        (late-degradation focus — state this in paper §3.3).

        Returns:
            X: (N, window_size, n_sensors)  float32
            y: (N,)                          float32
        """
        sequences, labels = [], []

        for i in range(n_trajectories):
            signals, rul = self._single_trajectory(seed_offset=i)
            T = len(signals)
            for start in range(0, T - self.window_size + 1):
                window_rul = rul[start + self.window_size - 1]
                if window_rul <= self.threshold:
                    sequences.append(signals[start : start + self.window_size])
                    labels.append(window_rul)

        if not sequences:
            return (
                np.empty((0, self.window_size, self.n_sensors), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels,    dtype=np.float32)

        # Shuffle so synthetic samples aren't in a block
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        return X[idx], y[idx]


def load_simulators(cfg: dict) -> dict[str, WeibullSimulator]:
    """
    Load calibrated simulators for all 4 clients from saved weibull_params.json.
    Call this inside FL client initialisation.
    """
    params_path = os.path.join(cfg["data"]["output_dir"], "weibull_params.json")
    with open(params_path) as f:
        params = json.load(f)

    return {
        fd: WeibullSimulator(
            k=params[fd]["k"],
            lam=params[fd]["lambda"],
            cfg=cfg,
            seed=cfg["reproducibility"]["seed"],
        )
        for fd in cfg["data"]["clients"]
    }