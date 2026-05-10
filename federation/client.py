"""
federation/client.py
====================
Flower federated learning client.

Each client:
  1. Loads its own preprocessed data partition (FD001–FD004)
  2. Optionally augments training data with Weibull synthetic trajectories
  3. Trains locally for N epochs using hybrid loss (or plain MSE)
  4. Returns updated parameters + metrics to the server

Supports three modes controlled by use_simulation / use_fedprox:
  Vanilla FedAvg:          use_simulation=False, use_fedprox=False
  Proposed (sim + SW-Agg): use_simulation=True,  use_fedprox=False
  FedProx baseline:        use_simulation=False,  use_fedprox=True

CHANGES vs original:
  - AMP (Automatic Mixed Precision) support via torch.amp.
    Enabled automatically when device is CUDA.  Gives ~1.5–2× speedup
    on RTX/T4 with negligible loss in RUL accuracy.
  - use_amp flag propagated through make_client_fn.
  - GradScaler created once per fit() call (safe for Flower simulation
    which may reuse the same PDMClient object across rounds).
"""

import json
import os

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml
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
<<<<<<< HEAD
        use_amp:        bool         = False,
        num_workers:    int          = 0,
=======
        use_amp:        bool         = True,
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd
    ):
        self.fd             = fd
        self.cfg            = cfg
        self.use_simulation = use_simulation
        self.use_fedprox    = use_fedprox
        self.device         = device
<<<<<<< HEAD
        self.use_amp        = use_amp
        self.num_workers    = num_workers
=======
        # AMP only makes sense on CUDA; silently disabled on CPU
        self.use_amp        = use_amp and (device.type == "cuda")
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd

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
                seed = cfg["reproducibility"]["seed"],
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
        FL round local training with AMP support.

        Steps:
          1. Receive and load global model weights from server.
          2. Snapshot global params for FedProx proximal term.
          3. Optionally augment local data with Weibull synthetic windows.
          4. Train for self.epochs with AdamW + cosine LR + AMP (if CUDA).
          5. Return updated weights + training metrics.
        """
        self.set_parameters(parameters)

        # Snapshot global params before any local update (for FedProx)
        global_params = [p.clone().detach() for p in self.model.parameters()]

        # Data augmentation
        if self.use_simulation:
            X_sim, y_sim = self.simulator.generate(
                n_trajectories=self.cfg["simulation"]["n_trajectories"]
            )
            X_tr = np.concatenate([self.X_train, X_sim], axis=0)
            y_tr = np.concatenate([self.y_train, y_sim], axis=0)
            rng  = np.random.default_rng(self.cfg["reproducibility"]["seed"])
            idx  = rng.permutation(len(X_tr))
            X_tr, y_tr = X_tr[idx], y_tr[idx]
        else:
            X_tr, y_tr = self.X_train, self.y_train

        loader = make_loader(
            X_tr, y_tr, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        # Local optimisation
<<<<<<< HEAD
        opt   = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        scaler = torch.amp.GradScaler(enabled=self.use_amp)
=======
        opt    = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched  = CosineAnnealingLR(opt, T_max=self.epochs)
        # GradScaler is created fresh each fit() call.
        # enabled=False is a no-op (no overhead) when use_amp is False,
        # so we can always construct it without an if-branch.
        scaler = GradScaler(enabled=self.use_amp)
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd
        self.model.train()

        total_loss = 0.0
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)

                opt.zero_grad()

<<<<<<< HEAD
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    rul_pred, hi_seq = self.model.forward_with_hi(X_batch)
                    loss, _          = self.criterion(rul_pred, y_batch, hi_seq)

                    if self.use_fedprox:
                        loss = self.fedprox_loss(loss, self.model, global_params)

=======
                # autocast is a no-op context when enabled=False
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    rul_pred, hi_seq = self.model.forward_with_hi(X_batch)
                    loss, _          = self.criterion(rul_pred, y_batch, hi_seq)
                    if self.use_fedprox:
                        loss = self.fedprox_loss(loss, self.model, global_params)

                # scaler.scale() is identity when enabled=False,
                # so the same code path works for both CPU and GPU.
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
<<<<<<< HEAD
=======

>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd
                epoch_loss += loss.item()

            sched.step()
            total_loss = epoch_loss / max(len(loader), 1)

        return (
            self.get_parameters({}),
            len(X_tr),
            {"train_loss": total_loss, "n_samples": len(X_tr), "fd": self.fd},
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
                # Use AMP for eval too — same hardware, free speedup
                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    preds.append(self.model(X.to(self.device)).cpu().float().numpy())
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
<<<<<<< HEAD
    use_amp:        bool = False,
    num_workers:    int  = 0,
=======
    use_amp:        bool = True,
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd
):
    """
    Returns a Flower-compatible client_fn(context: Context) → Client.

    Args:
        use_amp: enable AMP (passed through to PDMClient; auto-disabled on CPU)
    """
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
<<<<<<< HEAD
            num_workers    = num_workers,
        )
=======
        ).to_client()
>>>>>>> f71b955f3f917947bd21b6d6b3d0e5100c1934bd

    return client_fn