"""
models/loss.py
==============
Hybrid loss function.

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
        lambda_physics: weight of monotonicity term (default 0.1 from config)

    Usage in training loop:
        rul_pred, hi_seq = model.forward_with_hi(x)
        loss = criterion(rul_pred, rul_true, hi_seq)
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
            total_loss: scalar
            components: dict with 'mse', 'physics', 'total' for logging
        """
        # MSE on RUL
        mse = F.mse_loss(rul_pred, rul_true)

        # Monotonicity violation
        # hi_seq[:, t+1] should be >= hi_seq[:, t]
        # Penalise negative differences (decreases)
        hi_diff    = hi_seq[:, 1:] - hi_seq[:, :-1]          # (B, T-1)
        violation  = torch.relu(-hi_diff).mean()               # mean of decreases

        total = mse + self.lambda_physics * violation

        return total, {
            "mse":     mse.item(),
            "physics": violation.item(),
            "total":   total.item(),
        }


class PHMScore(nn.Module):
    """
    Asymmetric PHM competition scoring function.
    Penalises LATE predictions (positive error) more than EARLY predictions.

    S = Σ exp(-e_i/13) - 1    for e_i < 0  (early prediction)
      = Σ exp( e_i/10) - 1    for e_i >= 0 (late prediction)

    where e_i = RUL_pred_i - RUL_true_i

    Lower score = better. Report this alongside RMSE and MAE.
    """

    def forward(
        self, rul_pred: torch.Tensor, rul_true: torch.Tensor
    ) -> torch.Tensor:
        e = rul_pred - rul_true
        score = torch.where(
            e < 0,
            torch.exp(-e / 13.0) - 1,
            torch.exp( e / 10.0) - 1,
        )
        return score.sum()


class FedProxLoss(nn.Module):
    """
    FedProx proximal regularisation term.
    Added to the local loss to prevent excessive client drift.

    L_fedprox = L_local + (mu/2) * ||w - w_global||^2

    Reference: Li et al., "Federated Optimization in Heterogeneous
    Networks," ICLR 2020.
    """

    def __init__(self, mu: float = 0.01):
        super().__init__()
        self.mu = mu

    def proximal_term(
        self,
        local_model:  nn.Module,
        global_params: list[torch.Tensor],
    ) -> torch.Tensor:
        prox = torch.tensor(0.0, requires_grad=True)
        for local_p, global_p in zip(local_model.parameters(), global_params):
            prox = prox + torch.norm(local_p - global_p.detach()) ** 2
        return (self.mu / 2.0) * prox

    def forward(
        self,
        base_loss:    torch.Tensor,
        local_model:  nn.Module,
        global_params: list[torch.Tensor],
    ) -> torch.Tensor:
        return base_loss + self.proximal_term(local_model, global_params)


def build_criterion(cfg: dict) -> HybridRULLoss:
    return HybridRULLoss(lambda_physics=cfg["loss"]["lambda_physics"])