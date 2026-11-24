"""Custom quantitative loss functions for cross-sectional stock selection.

Key mathematical insight (Jensen's Inequality Volatility Trap):
  Naive exponential listwise ranking losses such as ListNet softmax(y / tau)
  optimize E[exp(y / tau) | x] ~ exp(mu / tau + sigma^2 / (2 * tau^2)). On
  heteroscedastic A-share returns where sigma^2 >> mu, the quadratic variance
  term dominates and forces the network to buy high-volatility lottery stocks.

  To align optimization with cross-sectional ranking WITHOUT volatility bias:
    1. PearsonICLoss(-corr(s, y_rank)) is strictly linear in the symmetric rank
       target y_rank, directly optimizing RankIC with zero variance bias.
    2. PairwiseRankNetLoss (Burges et al., 2005) compares symmetric random pairs
       (i, j) using sign(y_i - y_j), where P(y_i > y_j) > 0.5 iff mu_i > mu_j,
       eliminating heteroscedastic variance distortion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSectionalMSELoss(nn.Module):
    """Standard pointwise MSE on cross-sectionally standardized return targets."""

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(scores, targets)


class PearsonICLoss(nn.Module):
    """Negative cross-sectional Pearson correlation (-IC).

    When targets are cross-sectionally rank-normalized (y_rank), minimizing
    PearsonICLoss directly maximizes differentiable cross-sectional RankIC.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        s = scores - scores.mean()
        y = targets - targets.mean()
        cov = (s * y).sum()
        denom = torch.sqrt((s * s).sum() + self.eps) * torch.sqrt((y * y).sum() + self.eps)
        return -(cov / denom)


class PairwiseRankNetLoss(nn.Module):
    """Symmetric Pairwise RankNet loss (Burges et al., 2005) weighted by rank gap.

    For sampled pairs (i, j) with target difference dy = y_i - y_j:
      loss_ij = |dy| * log(1 + exp(-sign(dy) * (s_norm_i - s_norm_j)))
    Because both sign(dy) and |dy| are symmetric on bounded rank targets, high-
    volatility stocks gain zero artificial advantage over stable stocks.
    """

    def __init__(self, num_pairs: int = 2048, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_pairs = num_pairs
        self.eps = eps

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = scores.shape[0]
        if n < 2:
            return scores.sum() * 0.0

        s_norm = (scores - scores.mean()) / (scores.std(unbiased=False) + self.eps)
        idx_i = torch.randint(0, n, (self.num_pairs,), device=scores.device)
        idx_j = torch.randint(0, n, (self.num_pairs,), device=scores.device)

        dy = targets[idx_i] - targets[idx_j]
        valid = dy.abs() > 1e-6
        if not valid.any():
            return scores.sum() * 0.0

        dy = dy[valid]
        ds = s_norm[idx_i[valid]] - s_norm[idx_j[valid]]
        weights = dy.abs()
        weights = weights / (weights.mean() + self.eps)
        return (weights * F.softplus(-torch.sign(dy) * ds)).mean()


class CompositeICRankLoss(nn.Module):
    """Unbiased quantitative loss: -PearsonIC(s, y_rank) + lambda * PairwiseRankNet."""

    def __init__(self, ranknet_weight: float = 0.5, num_pairs: int = 2048) -> None:
        super().__init__()
        self.ic_loss = PearsonICLoss()
        self.rank_loss = PairwiseRankNetLoss(num_pairs=num_pairs)
        self.ranknet_weight = ranknet_weight

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ic_loss(scores, targets) + self.ranknet_weight * self.rank_loss(scores, targets)
