"""
models/loss.py
==============
Hybrid loss function — Novelty 3 of the paper.

  L_total = MSE(RUL_pred, RUL_true)
          + λ_physics * MonotonicityViolation(HI_sequence)

The monotonicity term penalises any timestep where HI decreases.
Physically, equipment cannot self-heal — HI must be non-decreasing.

Also contains:
  - PHMScore: the asymmetric competition metric (mandatory for CMAPSS papers)
  - FedProxLoss: proximal term used in the FedProx baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridRULLoss(nn.Module):
    """
    Physics-informed hybrid loss.

    Args:
        lambda_physics: weight of monotonicity constraint (from config)

    Usage in training loop:
        rul_pred, hi_seq = model.forward_with_hi(x)
        loss, components = criterion(rul_pred, rul_true, hi_seq)
    """

    def __init__(self, lambda_physics: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics

    def forward(
        self,
        rul_pred: torch.Tensor,   # (B,)
        rul_true: torch.Tensor,   # (B,)
        hi_seq:   torch.Tensor,   # (B, T) — HI over the window
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar tensor
            components: dict with 'mse', 'physics', 'total' for logging
        """
        # Data-driven term: MSE on RUL
        mse = F.mse_loss(rul_pred, rul_true)

        # Physics term: monotonicity violation
        # hi_seq[:, t+1] >= hi_seq[:, t]  must hold for all t.
        # Penalty = mean of all negative differences (decreases in HI).
        hi_diff   = hi_seq[:, 1:] - hi_seq[:, :-1]   # (B, T-1)
        violation = torch.relu(-hi_diff).mean()        # non-negative scalar

        total = mse + self.lambda_physics * violation

        return total, {
            "mse":     mse.item(),
            "physics": violation.item(),
            "total":   total.item(),
        }


class PHMScore(nn.Module):
    """
    Asymmetric PHM competition scoring function.

    Penalises LATE predictions (positive error) more heavily than early
    predictions to reflect the higher safety cost of missed failure warnings.

    S = Σ exp(-e_i / 13) − 1    for e_i <  0  (early — under-prediction)
      = Σ exp( e_i / 10) − 1    for e_i >= 0  (late  — over-prediction)

    where e_i = RUL_pred_i − RUL_true_i.
    Lower score = better. Report alongside RMSE and MAE.
    """

    def forward(
        self, rul_pred: torch.Tensor, rul_true: torch.Tensor
    ) -> torch.Tensor:
        e = rul_pred - rul_true
        score = torch.where(
            e < 0,
            torch.exp(-e / 13.0) - 1.0,
            torch.exp( e / 10.0) - 1.0,
        )
        return score.sum()


class FedProxLoss(nn.Module):
    """
    FedProx proximal regularisation.

    Adds a quadratic penalty that limits how far the local model drifts
    from the global model during each round of local training.

      L_fedprox = L_local + (μ/2) · Σ_l ||w_l − w̄_l||²

    where w̄_l are the global model parameters received at the start of
    the round (frozen — not updated during local training).

    Reference:
      Li et al., "Federated Optimization in Heterogeneous Networks
      (FedProx)," ICLR 2020.
    """

    def __init__(self, mu: float = 0.01):
        super().__init__()
        self.mu = mu

    def proximal_term(
        self,
        local_model:   nn.Module,
        global_params: list[torch.Tensor],
    ) -> torch.Tensor:
        device = next(local_model.parameters()).device
        # Start from a zero scalar on the correct device — no requires_grad
        # needed here; the gradient flows through local_p automatically.
        prox = torch.zeros(1, device=device)
        for local_p, global_p in zip(local_model.parameters(), global_params):
            diff  = local_p - global_p.detach().to(device)
            prox  = prox + torch.linalg.norm(diff) ** 2
        return (self.mu / 2.0) * prox

    def forward(
        self,
        base_loss:     torch.Tensor,
        local_model:   nn.Module,
        global_params: list[torch.Tensor],
    ) -> torch.Tensor:
        return base_loss + self.proximal_term(local_model, global_params)


def build_criterion(cfg: dict) -> HybridRULLoss:
    return HybridRULLoss(lambda_physics=cfg["loss"]["lambda_physics"])