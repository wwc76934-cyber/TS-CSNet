"""Offline unit tests for TS-CSNet anti-leakage, unbiased losses, models, and backtest engine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from tscs_alpha.backtest import run_soft_tilt_backtest, run_top_k_backtest
from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import PanelData
from tscs_alpha.losses import CompositeICRankLoss, PairwiseRankNetLoss, PearsonICLoss
from tscs_alpha.models import GRUScorer, MLPScorer, TSCSNet, count_parameters
from tscs_alpha.trainer import compute_rolling_splits


def _make_synthetic_panel(n_weeks: int = 260, n_stocks: int = 40) -> PanelData:
    """Create a tiny synthetic panel spanning 2015-2019 for fast offline tests."""
    dates = pd.bdate_range("2015-01-02", periods=n_weeks * 5 + 30).strftime("%Y-%m-%d").to_numpy(dtype="<U10")
    n_days = len(dates)
    sig_cal = np.arange(25, 25 + n_weeks * 5, 5, dtype=np.int32)
    exe_cal = sig_cal + 1
    sig_dates = dates[sig_cal]
    exe_dates = dates[exe_cal]

    lbl_end_cal = np.full(n_weeks, -1, dtype=np.int32)
    lbl_end_dates = np.full(n_weeks, "", dtype="<U10")
    for k in range(n_weeks - 2):
        lbl_end_cal[k] = exe_cal[k + 2]
        lbl_end_dates[k] = exe_dates[k + 2]

    rng = np.random.default_rng(7)
    hfq_close = np.cumprod(1.0 + rng.normal(0.0005, 0.015, size=(n_days, n_stocks)), axis=0).astype(np.float32) * 10.0
    hfq_open = (hfq_close * (1.0 + rng.normal(0.0, 0.003, size=(n_days, n_stocks)))).astype(np.float32)
    can_buy = np.ones((n_days, n_stocks), dtype=bool)
    can_sell = np.ones((n_days, n_stocks), dtype=bool)
    bench = np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=n_days))

    member_idx = np.tile(np.arange(n_stocks, dtype=np.int32), (n_weeks, 1))
    valid_mask = np.ones((n_weeks, n_stocks), dtype=bool)
    label_valid = np.ones((n_weeks, n_stocks), dtype=bool)
    label_valid[-2:, :] = False

    X_seq = rng.normal(0, 1, size=(n_weeks, n_stocks, 20, 6)).astype(np.float32)
    X_stat = rng.normal(0, 1, size=(n_weeks, n_stocks, 8)).astype(np.float32)
    X_tab = rng.normal(0, 1, size=(n_weeks, n_stocks, 26)).astype(np.float32)
    y_raw = rng.normal(0.01, 0.05, size=(n_weeks, n_stocks)).astype(np.float32)
    y_norm = (y_raw - y_raw.mean(axis=1, keepdims=True)) / (y_raw.std(axis=1, keepdims=True) + 1e-6)
    y_rank = np.zeros_like(y_norm)
    for k in range(n_weeks):
        r = pd.Series(y_raw[k]).rank().to_numpy(dtype=np.float32)
        y_rank[k] = (r - r.mean()) / (r.std() + 1e-6)

    return PanelData(
        signal_dates=sig_dates,
        exec_dates=exe_dates,
        label_end_dates=lbl_end_dates,
        signal_cal_idx=sig_cal,
        exec_cal_idx=exe_cal,
        label_end_cal_idx=lbl_end_cal,
        member_idx=member_idx,
        valid_mask=valid_mask,
        label_valid=label_valid,
        X_seq=X_seq,
        X_stat=X_stat,
        X_tab=X_tab,
        y_raw=y_raw,
        y_norm=y_norm,
        y_rank=y_rank,
        cal_dates=dates,
        stock_codes=np.array([f"sh.600{i:03d}" for i in range(n_stocks)]),
        hfq_open=hfq_open,
        hfq_close=hfq_close,
        can_buy=can_buy,
        can_sell=can_sell,
        bench_h00905=bench,
        bench_000905=bench,
    )


class TestTSCSAlpha(unittest.TestCase):
    def test_losses_scale_invariance_and_zero_volatility_bias(self) -> None:
        torch.manual_seed(0)
        y = torch.randn(200)
        y = (y - y.mean()) / y.std(unbiased=False)

        ic_loss = PearsonICLoss()
        rank_loss = PairwiseRankNetLoss(num_pairs=2048)
        comp_loss = CompositeICRankLoss(ranknet_weight=0.5, num_pairs=2048)

        # Perfect linear alignment gives IC = +1 -> loss = -1
        self.assertAlmostEqual(float(ic_loss(3.5 * y + 10.0, y)), -1.0, places=5)
        self.assertAlmostEqual(float(ic_loss(-2.0 * y + 5.0, y)), 1.0, places=5)
        self.assertLess(float(rank_loss(y, y)), float(rank_loss(-y, y)))
        self.assertLess(float(comp_loss(y, y)), float(comp_loss(-y, y)))

        # Zero heteroscedastic volatility bias check:
        # Two groups with identical mean 0 but different volatility (0.5 vs 3.0)
        vol = torch.cat([torch.full((100,), 0.5), torch.full((100,), 3.0)])
        loss_pos_vol = []
        loss_neg_vol = []
        for seed in range(100):
            torch.manual_seed(seed)
            y_het = torch.randn(200) * vol
            r = torch.argsort(torch.argsort(y_het)).float()
            y_r = (r - r.mean()) / (r.std(unbiased=False) + 1e-8)
            loss_pos_vol.append(float(comp_loss(vol, y_r)))
            loss_neg_vol.append(float(comp_loss(-vol, y_r)))
        # Symmetric loss should NOT systematically favor high-vol over low-vol when mean=0
        self.assertLess(abs(float(np.mean(loss_pos_vol)) - float(np.mean(loss_neg_vol))), 0.05)

    def test_models_forward_backward_and_param_budget(self) -> None:
        torch.manual_seed(1)
        x_seq = torch.randn(64, 20, 6)
        x_stat = torch.randn(64, 8)
        x_tab = torch.randn(64, 26)
        y = torch.randn(64)
        loss_fn = CompositeICRankLoss()

        for model in (MLPScorer(in_dim=26), GRUScorer(seq_dim=6, static_dim=8), TSCSNet(seq_dim=6, static_dim=8)):
            n_p = count_parameters(model)
            self.assertGreater(n_p, 1_000)
            self.assertLess(n_p, 25_000)

            out, gate = model(x_seq, x_stat, x_tab, return_gate=True)
            self.assertEqual(tuple(out.shape), (64,))
            if isinstance(model, TSCSNet):
                self.assertTrue(0.0 < gate < 1.0)

            loss = loss_fn(out, y)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertEqual(len(grads), len(list(model.parameters())))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))

    def test_strict_purge_gap_in_rolling_splits(self) -> None:
        panel = _make_synthetic_panel(n_weeks=260, n_stocks=60)
        for year in (2017, 2018, 2019):
            tr, va, te = compute_rolling_splits(panel, test_year=year, train_years_window=3, val_weeks=20)
            first_test_sig = str(panel.signal_dates[te[0]])
            first_val_sig = str(panel.signal_dates[va[0]])

            for k in va:
                self.assertLessEqual(str(panel.label_end_dates[k]), first_test_sig)
            for k in tr:
                self.assertLessEqual(str(panel.label_end_dates[k]), first_val_sig)

    def test_backtest_limit_locks_and_buffer_behavior(self) -> None:
        panel = _make_synthetic_panel(n_weeks=200, n_stocks=30)
        cfg = ExperimentConfig(
            test_years=(2017, 2018),
            top_k=5,
            hold_buffer=10,
            rebalance_stride=2,
        )
        base_scores = np.tile(np.linspace(3.0, -3.0, 30, dtype=np.float32), (len(panel.signal_dates), 1))

        rebal_k = [k for k, d in enumerate(panel.signal_dates) if str(d)[:4] in {"2017", "2018"}][::2]
        d2 = int(panel.exec_cal_idx[rebal_k[1]])
        panel.can_sell[d2, 0] = False
        panel.can_buy[d2, 5] = False

        res = run_top_k_backtest("TestStrat", panel, base_scores, cfg)
        self.assertTrue(np.isfinite(res.nav).all())
        self.assertTrue((res.nav > 0).all())
        self.assertLess(np.mean(res.rebal_turnovers[1:]), 0.15)

        res_tilt = run_soft_tilt_backtest("TestTilt", panel, base_scores, cfg, mv_scale=0.0)
        self.assertTrue(np.isfinite(res_tilt.nav).all())
        self.assertTrue((res_tilt.nav > 0).all())
        self.assertLess(np.mean(res_tilt.rebal_turnovers[1:]), 0.15)


if __name__ == "__main__":
    unittest.main()
