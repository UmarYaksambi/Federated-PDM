"""
models/tcn.py
=============
Temporal Convolutional Network (TCN) for RUL estimation.

Architecture:
  Input  →  4x Dilated Residual Blocks (dilation 1,2,4,8)
         →  Multi-head Attention over temporal dimension
         →  FC head → RUL scalar

MC-Dropout:
  Dropout stays ACTIVE at inference time (model.train() mode).
  Run N forward passes → mean = point estimate, std = uncertainty.
  See evaluate.py::mc_predict() for usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        # Causal padding: pad only the left so future info is never seen
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      dilation=dilation, padding=pad)
        )
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      dilation=dilation, padding=pad)
        )
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

        # 1x1 conv to match channel dims when in != out
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self._pad = pad

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove the right-side padding introduced by causal conv."""
        return x[:, :, : -self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch
        out = self.relu(self._chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._chomp(self.conv2(out)))
        out = self.dropout(out)
        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the time dimension.
    Input:  (B, C, T)  — channels-first from TCN
    Output: (B, C)     — attended global representation
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C) for attention
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Global average pool over time → (B, C)
        return x.mean(dim=1)


class TCN(nn.Module):
    """
    Uncertainty-Aware Temporal Convolutional Network.

    Args:
        input_size:      number of sensor features (14 for CMAPSS)
        hidden_channels: channels in each residual block
        num_levels:      number of dilated blocks (dilation doubles each level)
        kernel_size:     conv kernel size
        dropout:         dropout rate (used for MC-Dropout at inference too)
    """

    def __init__(
        self,
        input_size:      int   = 14,
        hidden_channels: int   = 64,
        num_levels:      int   = 4,
        kernel_size:     int   = 3,
        dropout:         float = 0.2,
    ):
        super().__init__()

        # TCN stack
        blocks = []
        for i in range(num_levels):
            in_ch  = input_size if i == 0 else hidden_channels
            dil    = 2 ** i
            blocks.append(
                _ResidualBlock(in_ch, hidden_channels, kernel_size, dil, dropout)
            )
        self.tcn = nn.Sequential(*blocks)

        # Temporal attention
        self.attention = _TemporalAttention(hidden_channels, num_heads=4)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),   # MC-Dropout: keep active at test time
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)  — batch, timesteps, features
        Returns:
            rul_pred: (B,)  — RUL point estimate
        """
        # TCN expects channels-first: (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)          # (B, hidden, T)
        x = self.attention(x)    # (B, hidden)
        return self.head(x).squeeze(-1)   # (B,)

    def forward_with_hi(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns both RUL prediction AND an intermediate HI sequence.
        Used in training so the physics loss can access the HI sequence.

        HI sequence: sigmoid of per-timestep feature mean — monotonicity
        is then enforced via the physics consistency loss, not the architecture.
        """
        x_perm = x.permute(0, 2, 1)       # (B, F, T)
        feat   = self.tcn(x_perm)          # (B, hidden, T)
        # HI proxy: sigmoid of mean activation across channels, per timestep
        hi_seq = torch.sigmoid(feat.mean(dim=1))   # (B, T)
        attended = self.attention(feat)            # (B, hidden)
        rul_pred = self.head(attended).squeeze(-1) # (B,)
        return rul_pred, hi_seq


def build_model(cfg: dict) -> TCN:
    """Construct TCN from config.yaml values."""
    m = cfg["model"]
    return TCN(
        input_size      = m["input_size"],
        hidden_channels = m["hidden_channels"],
        num_levels      = m["num_levels"],
        kernel_size     = m["kernel_size"],
        dropout         = m["dropout"],
    )