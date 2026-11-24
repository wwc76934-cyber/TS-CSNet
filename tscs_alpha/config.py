"""Central configuration for the TS-CSNet CSI 500 stock selection project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    # Paths
    project_root: Path = Path(r"D:\QuantProjects\tscs-alpha")
    source_data_dir: Path = Path(r"D:\QuantProjects\csi500-alpha\data\raw")

    @property
    def cache_path(self) -> Path:
        return self.project_root / "data" / "panel_cache.npz"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figure_dir(self) -> Path:
        return self.project_root / "figures"

    # Data & Feature Specs (6 daily sequence channels + 8 pure-PIT static features)
    seq_len: int = 20
    seq_features: Tuple[str, ...] = (
        "ret_1d",        # Daily close-to-close return
        "gap_oc",        # Overnight open/prev_close log gap
        "log_hl",        # Intraday log(high / low) amplitude
        "close_pos",     # Relative close position inside [low, high]
        "log_turn",      # log1p(turnover rate)
        "amt_ratio",     # log(amount / trailing_20d_avg_amount)
    )
    static_features: Tuple[str, ...] = (
        "ep_ttm",        # 1 / PE_TTM (strict BaoStock announcement-day PIT)
        "bp_mrq",        # 1 / PB_MRQ (strict PIT)
        "roe_ttm",       # PB_MRQ / PE_TTM implied ROE_TTM (strict PIT)
        "roe_chg_60d",   # 60-day change in implied ROE_TTM (pure-PIT earnings improvement)
        "log_float_mv",  # log(float market cap)
        "mom_60d",       # Trailing 60-day return
        "vol_60d",       # Trailing 60-day daily return volatility
        "turn_cv_20d",   # 20-day turnover coefficient of variation std(turn)/mean(turn)
    )

    # Prediction & Rebalance Horizon
    # Weekly schedule step = 1 (~5 trading days); biweekly horizon = 2 schedule steps (~10 trading days)
    horizon_steps: int = 2
    rebalance_stride: int = 2

    # Rolling Out-of-Sample Protocol (strictly isolating 2026+)
    test_years: Tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
    train_years_window: int = 3   # Rolling 3-year lookback (train + val)
    val_weeks: int = 24           # Last ~6 months inside lookback used for early stopping

    # Neural Architecture Hyperparameters (lightweight & CPU-friendly: ~15k params)
    hidden_dim: int = 32
    num_heads: int = 2
    dropout: float = 0.15

    # Loss & Optimization (unbiased PearsonIC + Symmetric Pairwise RankNet on rank targets)
    ranknet_weight: float = 0.5
    ranknet_pairs: int = 2048
    lr: float = 2.5e-3
    weight_decay: float = 1e-2
    epochs: int = 20
    patience: int = 5
    grad_accum_dates: int = 4
    grad_clip: float = 1.0
    seeds: Tuple[int, ...] = (42, 102, 202)
    ridge_alpha: float = 500.0

    # Portfolio Construction (clean Top-K + hysteresis buffer + causal EMA score smoothing)
    top_k: int = 50
    hold_buffer: int = 100
    signal_ema_alpha: float = 0.5   # Causal exponential smoothing across biweekly scores

    # A-Share Realistic Transaction Costs (buy: 3bp comm + 7bp slip; sell: stamp + 3bp comm + 7bp slip)
    buy_fee_rate: float = 0.0010
    sell_fee_before_stamp_cut: float = 0.0020  # Before 2023-08-28 (0.10% stamp duty)
    sell_fee_after_stamp_cut: float = 0.0015   # From 2023-08-28 (0.05% stamp duty)
    stamp_cut_date: str = "2023-08-28"
    stress_single_side_fee: float = 0.0020     # Flat 20bp single-side stress test

    # Model Registry (ordered progressive ablation ladder)
    model_names: Tuple[str, ...] = (
        "M1_Ridge",
        "M2_MLP_MSE",
        "M3_GRU_MSE",
        "M4_GRU_ICRank",
        "M5_TSCSNet",
    )
