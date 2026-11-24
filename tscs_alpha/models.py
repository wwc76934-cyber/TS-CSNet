"""Self-built PyTorch architectures for progressive ablation on CSI 500 selection.

Models share a lightweight hidden dimension (d=32, ~12k-20k parameters) so that
9-year rolling cross-sectional training runs cleanly on pure CPU while avoiding
overfitting on low-SNR financial data:
  - MLPScorer : Vanilla tabular deep baseline (M2_MLP_MSE)
  - GRUScorer : Temporal GRU + static concat baseline (M3_GRU_MSE, M4_GRU_ICRank)
  - TSCSNet   : Temporal GRU + Gated Fusion + Cross-Sectional Self-Attention (M5_TSCSNet)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MLPScorer(nn.Module):
    """Vanilla 2-layer MLP operating on summarized tabular features (N, F_tab)."""

    def __init__(self, in_dim: int = 24, hidden_dim: int = 32, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_stat: torch.Tensor,
        x_tab: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        scores = self.net(x_tab).squeeze(-1)
        if return_gate:
            return scores, float("nan")
        return scores


class GRUScorer(nn.Module):
    """Independent-stock temporal GRU encoder fused with static PIT context."""

    def __init__(
        self,
        seq_dim: int = 6,
        static_dim: int = 6,
        hidden_dim: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.seq_norm = nn.LayerNorm(hidden_dim)
        self.stat_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_stat: torch.Tensor,
        x_tab: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        # x_seq: (N, T, C_seq), x_stat: (N, C_stat)
        _, h_last = self.gru(x_seq)          # (1, N, d)
        h_seq = self.seq_norm(h_last[-1])    # (N, d)
        h_stat = self.stat_proj(x_stat)      # (N, d)
        scores = self.fuse(torch.cat([h_seq, h_stat], dim=-1)).squeeze(-1)
        if return_gate:
            return scores, float("nan")
        return scores


class TSCSNet(nn.Module):
    """Temporal-CrossSectional Network with Gated Fusion & Cross-Sectional Attention.

    Architecture for a single cross-section of N active stocks on date t:
      1. Temporal GRU encodes each stock's (T=20, C=6) price-volume sequence -> h_seq.
      2. Static Encoder projects 6 pure-PIT fundamental/medium-term features -> h_stat.
      3. Gated Fusion blends h_seq and h_stat via a learnable sigmoid gate g in (0, 1).
      4. Cross-Sectional Multi-Head Self-Attention lets all N stocks interact on date t
         so each stock's final representation reflects relative cross-sectional strength.
      5. Linear head outputs cross-sectional alpha scores.
    """

    def __init__(
        self,
        seq_dim: int = 6,
        static_dim: int = 6,
        hidden_dim: int = 32,
        num_heads: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.seq_norm = nn.LayerNorm(hidden_dim)

        self.stat_proj = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable gate balancing short-term sequence dynamics vs static fundamentals
        self.gate_layer = nn.Linear(hidden_dim * 2, hidden_dim)

        # Pre-Norm Cross-Sectional Self-Attention block across the N stocks on date t
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.cs_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # Lightweight Pre-Norm Feed-Forward block
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_stat: torch.Tensor,
        x_tab: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        # 1. Temporal sequence encoding
        _, h_last = self.gru(x_seq)               # (1, N, d)
        h_seq = self.seq_norm(h_last[-1])         # (N, d)

        # 2. Static PIT encoding
        h_stat = self.stat_proj(x_stat)           # (N, d)

        # 3. Gated fusion (g -> 1 favors temporal price-volume, g -> 0 favors static PIT)
        gate = torch.sigmoid(self.gate_layer(torch.cat([h_seq, h_stat], dim=-1)))
        h = gate * h_seq + (1.0 - gate) * h_stat  # (N, d)

        # 4. Cross-sectional self-attention across all N stocks in the same trading date
        h_in = self.attn_norm(h).unsqueeze(0)     # (1, N, d)
        attn_out, _ = self.cs_attn(h_in, h_in, h_in, need_weights=False)
        h = h + self.attn_drop(attn_out.squeeze(0))
        h = h + self.ffn(self.ffn_norm(h))

        # 5. Cross-sectional score projection
        scores = self.head(self.head_norm(h)).squeeze(-1)
        if return_gate:
            return scores, float(gate.mean().item())
        return scores


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
