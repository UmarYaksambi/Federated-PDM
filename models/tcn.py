"""
models/tcn.py
=============
Temporal Convolutional Network (TCN) for RUL estimation.

Architecture:
  Input  →  4x Dilated Residual Blocks (dilation 1,2,4,8)
         →  Multi-head Temporal Self-Attention
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
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float,
    ):
        super().__init__()
        # Causal convolution: pad only the left so the model never sees
        # future timesteps — mandatory for autoregressive-style TCNs.
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
        self._pad    = pad

        # 1×1 projection when channel dims change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )

    def _chomp(self, x: torch.Tensor) -> torch.Tensor:
        """Remove right-side padding to maintain causal property."""
        return x[:, :, : -self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self._chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._chomp(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class _TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.
    Input:  (B, C, T)  — channels-first from TCN output
    Output: (B, C)     — attended global representation (pooled over time)
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
        # (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        # Global average pool over time → (B, C)
        return x.mean(dim=1)


class TCN(nn.Module):
    """
    Uncertainty-Aware Temporal Convolutional Network for RUL estimation.

    Args:
        input_size:      number of sensor features (14 for CMAPSS)
        hidden_channels: feature channels per residual block
        num_levels:      number of dilated blocks (dilation 1, 2, 4, 8, ...)
        kernel_size:     temporal conv kernel size
        dropout:         applied during training AND MC-Dropout inference
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
            in_ch = input_size if i == 0 else hidden_channels
            blocks.append(
                _ResidualBlock(in_ch, hidden_channels, kernel_size,
                               dilation=2 ** i, dropout=dropout)
            )
        self.tcn = nn.Sequential(*blocks)

        # Temporal attention
        self.attention = _TemporalAttention(hidden_channels, num_heads=4)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),   # kept active at test time for MC-Dropout
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialise weights. For weight_norm-wrapped Conv1d layers, the
        actual trainable tensors are weight_v (direction) and weight_g
        (magnitude). Calling kaiming_normal_ on m.weight (which is a
        computed property: weight_g * weight_v / ||weight_v||) has no effect
        on the underlying parameters. We target weight_v directly.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if hasattr(m, "weight_v"):
                    # weight_norm is active — init the direction vector
                    nn.init.kaiming_normal_(m.weight_v, nonlinearity="relu")
                elif hasattr(m, "weight"):
                    # plain Conv1d (e.g., 1×1 downsample projection)
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) — batch, timesteps, sensor features
        Returns:
            rul_pred: (B,) — scalar RUL estimate
        """
        x = x.permute(0, 2, 1)          # → (B, F, T)
        x = self.tcn(x)                  # → (B, hidden, T)
        x = self.attention(x)            # → (B, hidden)
        return self.head(x).squeeze(-1)  # → (B,)

    def forward_with_hi(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns RUL prediction AND a per-timestep Health Index sequence.
        The HI sequence is used by the physics consistency loss (loss.py)
        to penalise any decrease in HI over the window.

        HI proxy: sigmoid of the mean channel activation at each timestep.
        Monotonicity is NOT enforced architecturally here — it is enforced
        softly through the loss function.
        """
        x_perm   = x.permute(0, 2, 1)               # (B, F, T)
        feat     = self.tcn(x_perm)                  # (B, hidden, T)
        hi_seq   = torch.sigmoid(feat.mean(dim=1))   # (B, T)
        attended = self.attention(feat)               # (B, hidden)
        rul_pred = self.head(attended).squeeze(-1)   # (B,)
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