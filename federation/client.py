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
"""

import json
import os

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import yaml
from flwr.common import Context          # FIX 4: needed for updated client_fn signature
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
    """

    def __init__(
        self,
        fd:             str,
        cfg:            dict,
        use_simulation: bool         = True,
        use_fedprox:    bool         = False,
        device:         torch.device = torch.device("cpu"),
    ):
        self.fd             = fd
        self.cfg            = cfg
        self.use_simulation = use_simulation
        self.use_fedprox    = use_fedprox
        self.device         = device

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
              f"fedprox={'on' if use_fedprox else 'off'}")

    # ------------------------------------------------------------------ #
    # Flower interface
    # ------------------------------------------------------------------ #

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        # FIX 1 + 2: use state_dict().values() instead of model.parameters().
        #
        # model.parameters() only yields *trainable* tensors and still carries
        # requires_grad=True, so calling .numpy() on them raises:
        #   RuntimeError: Can't call numpy() on Tensor that requires grad.
        #
        # state_dict() includes ALL tensors (parameters + buffers such as
        # BatchNorm running stats or weight_norm pre-hook tensors) and returns
        # them already detached from the autograd graph, so .numpy() works
        # directly.  Using state_dict() here keeps get/set_parameters and
        # _load_final_model in train_federated.py consistent with each other.
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        # FIX 2 (continued): zip against state_dict keys — matches get_parameters.
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
          2. Snapshot global params for FedProx proximal term.
          3. Optionally augment local data with Weibull synthetic windows.
          4. Train for self.epochs with AdamW + cosine LR schedule.
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

        loader = make_loader(X_tr, y_tr, self.batch_size, shuffle=True)

        # Local optimisation
        opt   = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        self.model.train()

        total_loss = 0.0
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                opt.zero_grad()
                rul_pred, hi_seq = self.model.forward_with_hi(X_batch)
                loss, _          = self.criterion(rul_pred, y_batch, hi_seq)

                if self.use_fedprox:
                    loss = self.fedprox_loss(loss, self.model, global_params)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

            sched.step()
            total_loss = epoch_loss / max(len(loader), 1)

        return (
            self.get_parameters({}),
            len(X_tr),
            # "fd" is included so SimilarityWeightedStrategy.aggregate_fit()
            # can look up the correct per-client weight without relying on
            # proxy.cid, which is a large hash in Flower >= 1.x simulation
            # mode and is NOT a sequential 0..N-1 index.
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
                preds.append(self.model(X.to(self.device)).cpu().numpy())
                trues.append(y.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        m     = compute_metrics(preds, trues)

        # Flower requires (loss, num_examples, metrics_dict)
        return float(m["rmse"]), len(self.X_test), m


# ------------------------------------------------------------------ #
# Client factory for Flower simulation
# ------------------------------------------------------------------ #

def make_client_fn(
    cfg:            dict,
    use_simulation: bool,
    use_fedprox:    bool,
    device:         torch.device,
):
    """
    Returns a Flower-compatible client_fn(context: Context) → Client.

    FIX 4: Flower's newer versions expect:
      - client_fn(context: Context) signature (not cid: str)
      - return type Client, not NumPyClient
        (call .to_client() to convert NumPyClient → Client)

    In simulation mode the virtual-client index is available via
    context.node_config["partition-id"] when Flower sets it, with a
    fallback to context.node_id for older builds.
    """
    fd_keys = cfg["data"]["clients"]

    def client_fn(context: Context) -> fl.client.Client:
        # partition-id is set by Flower's VCE in newer releases;
        # node_id (int) is the reliable fallback for start_simulation().
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
        ).to_client()   # NumPyClient → Client (required by current Flower)

    return client_fn