"""
federation/client.py
====================
Flower federated learning client.

Each client:
  1. Loads its own preprocessed data partition (FD001–FD004)
  2. Optionally augments training data with Weibull synthetic trajectories
  3. Trains locally for N epochs using hybrid loss
  4. Returns updated parameters + metrics to the server

FIXES IN THIS VERSION
=====================
Fix 1 — Round-specific augmentation seed
  Original code: rng = np.random.default_rng(cfg["seed"])
  Problem: identical shuffle every round → clients see identical data
  order each round, reinforcing the same local bias.
  Fixed: seed = base_seed + round_num * large_prime, giving a unique
  permutation each round while still being reproducible.

Fix 2 — Inter-round learning rate decay (cosine across communication rounds)
  Within each round the local optimiser already uses a cosine schedule
  (CosineAnnealingLR over local epochs). But there is no decay across
  the 50 communication rounds. Without it, clients take equally large
  steps in round 50 as in round 1, driving client drift even as the
  global model converges.

  The server now passes {"round": r, "num_rounds": R} in fit_config.
  We compute:
    cos_factor   = 0.5 · (1 + cos(π · (r−1) / (R−1)))   ∈ [0, 1]
    effective_lr = base_lr · (min_frac + (1−min_frac) · cos_factor)
  where min_frac = 0.1, so LR decays from base_lr to 0.1·base_lr.

AMP (Automatic Mixed Precision)
  Enabled automatically when device is CUDA.  Gives ~1.5–2× speedup.
"""

import json
import math
import os

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import Context
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluate import compute_metrics, make_loader
from models.loss import FedProxLoss, HybridRULLoss
from models.tcn import build_model
from simulation.weibull import WeibullSimulator


class PDMClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for federated predictive maintenance.

    Args:
        fd:              sub-dataset identifier, e.g. "FD001"
        cfg:             parsed config.yaml dict
        use_simulation:  augment local training data with Weibull synthesis
        use_fedprox:     add FedProx proximal term to local loss
        device:          torch device for this client
        use_amp:         enable Automatic Mixed Precision (CUDA only)
    """

    def __init__(
        self,
        fd:             str,
        cfg:            dict,
        use_simulation: bool         = True,
        use_fedprox:    bool         = False,
        device:         torch.device = torch.device("cpu"),
        use_amp:        bool         = True,
    ):
        self.fd             = fd
        self.cfg            = cfg
        self.use_simulation = use_simulation
        self.use_fedprox    = use_fedprox
        self.device         = device
        self.use_amp        = use_amp and (device.type == "cuda")
        self._base_seed     = cfg["reproducibility"]["seed"]

        train_cfg       = cfg["training"]
        self.epochs     = train_cfg["epochs_local"]
        self.batch_size = train_cfg["batch_size"]
        self.lr         = train_cfg["learning_rate"]
        self.wd         = train_cfg["weight_decay"]

        # Load pre-processed partition
        npz = np.load(os.path.join(cfg["data"]["output_dir"], f"{fd}.npz"))
        self.X_train = npz["X_train"]
        self.y_train = npz["y_train"]
        self.X_test  = npz["X_test"]
        self.y_test  = npz["y_test"]

        # Weibull simulator (only when simulation is enabled)
        if use_simulation:
            params_path = os.path.join(
                cfg["data"]["output_dir"], "weibull_params.json"
            )
            with open(params_path) as f:
                params = json.load(f)
            self.simulator = WeibullSimulator(
                k    = params[fd]["k"],
                lam  = params[fd]["lambda"],
                cfg  = cfg,
                seed = self._base_seed,
            )

        # Loss functions
        self.criterion    = HybridRULLoss(
            lambda_physics=cfg["loss"]["lambda_physics"]
        )
        self.fedprox_loss = FedProxLoss(mu=cfg["federation"]["fedprox_mu"])

        # Model — weights will be set by server before each round
        self.model = build_model(cfg).to(device)

        print(f"  Client {fd} | "
              f"train={len(self.X_train):,}  test={len(self.X_test):,}  "
              f"sim={'on' if use_simulation else 'off'}  "
              f"fedprox={'on' if use_fedprox else 'off'}  "
              f"amp={'on' if self.use_amp else 'off'}")

    # ------------------------------------------------------------------ #
    # Flower interface
    # ------------------------------------------------------------------ #

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        new_state = {
            k: torch.tensor(v, dtype=torch.float32).to(self.device)
            for k, v in params_dict
        }
        self.model.load_state_dict(new_state, strict=True)

    def fit(
        self,
        parameters: list[np.ndarray],
        config:     dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """
        FL round local training.

        Steps:
          1. Receive and load global model weights from server.
          2. Read round / num_rounds from server config for LR scheduling.
          3. Snapshot global params for FedProx proximal term.
          4. Optionally augment local data (round-specific seed — Fix 1).
          5. Train with AdamW + inter-round LR decay (Fix 2) + AMP.
          6. Return updated weights + training metrics.
        """
        self.set_parameters(parameters)

        # ---- Fix 2: inter-round cosine LR decay ----
        round_num    = int(config.get("round",      1))
        total_rounds = int(config.get("num_rounds",
                           self.cfg["federation"].get("num_rounds", 50)))

        # Cosine decay: round 1 → base_lr, final round → 0.1 * base_lr
        cos_factor   = 0.5 * (
            1.0 + math.cos(math.pi * (round_num - 1) / max(total_rounds - 1, 1))
        )
        effective_lr = self.lr * (0.1 + 0.9 * cos_factor)

        # Snapshot global params before any local update (for FedProx)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # ---- Data augmentation ----
        if self.use_simulation:
            X_sim, y_sim = self.simulator.generate(
                n_trajectories=self.cfg["simulation"]["n_trajectories"]
            )
            X_tr = np.concatenate([self.X_train, X_sim], axis=0)
            y_tr = np.concatenate([self.y_train, y_sim], axis=0)

            # Fix 1: unique permutation each round, still reproducible
            # Large prime multiplier spreads seeds well across rounds.
            rng = np.random.default_rng(self._base_seed + round_num * 7_919)
            idx = rng.permutation(len(X_tr))
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        else:
            X_tr, y_tr = self.X_train, self.y_train

        loader = make_loader(X_tr, y_tr, self.batch_size, shuffle=True)

        # ---- Local optimisation ----
        opt    = AdamW(self.model.parameters(), lr=effective_lr, weight_decay=self.wd)
        sched  = CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = GradScaler(enabled=self.use_amp)
        self.model.train()

        total_loss = 0.0
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                opt.zero_grad()

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    rul_pred, hi_seq = self.model.forward_with_hi(X_batch)
                    loss, _          = self.criterion(rul_pred, y_batch, hi_seq)
                    if self.use_fedprox:
                        loss = self.fedprox_loss(loss, self.model, global_params)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()

                epoch_loss += loss.item()

            sched.step()
            total_loss = epoch_loss / max(len(loader), 1)

        return (
            self.get_parameters({}),
            len(X_tr),
            {
                "train_loss": total_loss,
                "n_samples":  len(X_tr),
                "fd":         self.fd,
                "lr":         effective_lr,   # visible in convergence CSV
            },
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config:     dict,
    ) -> tuple[float, int, dict]:
        """Per-round server-side evaluation called by Flower."""
        self.set_parameters(parameters)
        self.model.eval()

        loader = make_loader(self.X_test, self.y_test, self.batch_size)
        preds, trues = [], []
        with torch.no_grad():
            for X, y in loader:
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds.append(
                        self.model(X.to(self.device)).cpu().float().numpy()
                    )
                trues.append(y.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        m     = compute_metrics(preds, trues)

        return float(m["rmse"]), len(self.X_test), m


# ------------------------------------------------------------------ #
# Client factory for Flower simulation
# ------------------------------------------------------------------ #

def make_client_fn(
    cfg:            dict,
    use_simulation: bool,
    use_fedprox:    bool,
    device:         torch.device,
    use_amp:        bool = True,
):
    """Returns a Flower-compatible client_fn(context: Context) → Client."""
    fd_keys = cfg["data"]["clients"]

    def client_fn(context: Context) -> fl.client.Client:
        partition_id = context.node_config.get(
            "partition-id", int(context.node_id)
        )
        fd = fd_keys[int(partition_id) % len(fd_keys)]
        return PDMClient(
            fd             = fd,
            cfg            = cfg,
            use_simulation = use_simulation,
            use_fedprox    = use_fedprox,
            device         = device,
            use_amp        = use_amp,
        ).to_client()

    return client_fn