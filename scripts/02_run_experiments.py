"""Step 2: Run full 9-year (2017-2025) rolling out-of-sample ablation experiments.

Supports `--eval-only` to re-run backtest evaluation directly from saved
`outputs/oos_scores.npz` without retraining.
Evaluates both:
  1. Primary: Soft-Tilt Index Enhancement (benchmark-aligned float-cap base +
     score tilt, zeroing out bottom-tail losers across all 500 constituents).
  2. Comparison: Concentrated Top-50 / Buffer-100 equal-weight stock picking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tscs_alpha.backtest import (
    BacktestResult,
    evaluate_signal_ic,
    get_oos_rebalance_indices,
    run_soft_tilt_backtest,
    run_top_k_backtest,
)
from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import load_panel
from tscs_alpha.models import GRUScorer, MLPScorer, TSCSNet, count_parameters
from tscs_alpha.trainer import ModelRunResult, run_neural_rolling, run_ridge_rolling


def _param_count_for(model_name: str, cfg: ExperimentConfig) -> int:
    seq_dim = len(cfg.seq_features)
    stat_dim = len(cfg.static_features)
    tab_dim = stat_dim + 3 * seq_dim
    if model_name == "M1_Ridge":
        return tab_dim
    if model_name == "M2_MLP_MSE":
        return count_parameters(MLPScorer(in_dim=tab_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout))
    if model_name in ("M3_GRU_MSE", "M4_GRU_ICRank"):
        return count_parameters(GRUScorer(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout))
    if model_name == "M5_TSCSNet":
        return count_parameters(TSCSNet(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, dropout=cfg.dropout))
    return 0


def _b0_row(b0_nav: np.ndarray) -> Dict[str, float | str]:
    n_years = len(b0_nav) / 242.0
    b0_ext = np.concatenate([[1.0], b0_nav])
    b0_ret = b0_ext[1:] / b0_ext[:-1] - 1.0
    b0_ann = float(b0_nav[-1] ** (1.0 / n_years) - 1.0)
    b0_vol = float(np.std(b0_ret, ddof=1) * np.sqrt(242.0))
    b0_peak = np.maximum.accumulate(b0_nav)
    b0_mdd = float((b0_nav / b0_peak - 1.0).min())
    return {
        "Model": "B0_CSI500_TR",
        "Params": 0,
        "RankIC": np.nan,
        "RankICIR_Ann": np.nan,
        "RankIC_WinRate": np.nan,
        "Top50_Biweekly_Excess": np.nan,
        "TopMinusBot50_Biweekly": np.nan,
        "Ann_Return": b0_ann,
        "Ann_Excess": 0.0,
        "IR": 0.0,
        "Sharpe": b0_ann / (b0_vol + 1e-12),
        "MaxDD": b0_mdd,
        "Rel_MaxDD": 0.0,
        "Ann_OneWay_Turnover": 0.0,
        "Yearly_Excess_WinRate": np.nan,
        "Stress20bp_Ann_Excess": 0.0,
        "Stress20bp_IR": 0.0,
        "Train_Sec": 0.0,
    }


def main() -> None:
    cfg = ExperimentConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    eval_only = "--eval-only" in sys.argv

    panel = load_panel(cfg)
    rebal_indices = get_oos_rebalance_indices(panel, cfg)
    print(f"Loaded panel: {len(panel.signal_dates)} weeks, OOS biweekly rebalances={len(rebal_indices)}")
    t_all = time.time()

    run_results: Dict[str, ModelRunResult] = {}

    if eval_only:
        saved = np.load(cfg.output_dir / "oos_scores.npz")
        val_ic_path = cfg.output_dir / "yearly_val_ic.json"
        val_ic_map = json.loads(val_ic_path.read_text(encoding="utf-8")) if val_ic_path.exists() else {}
        for m_name in cfg.model_names:
            run_results[m_name] = ModelRunResult(
                model_name=m_name,
                num_params=_param_count_for(m_name, cfg),
                oos_scores=saved[m_name],
                gate_history={},
                yearly_val_ic={int(k): float(v) for k, v in val_ic_map.get(m_name, {}).items()},
                elapsed_seconds=0.0,
            )
    else:
        for m_name in cfg.model_names:
            print(f"\n--- Running {m_name} ---")
            if m_name == "M1_Ridge":
                res = run_ridge_rolling(panel, cfg)
            else:
                res = run_neural_rolling(m_name, panel, cfg, verbose=True)
            run_results[m_name] = res

        np.savez_compressed(
            cfg.output_dir / "oos_scores.npz",
            **{m: r.oos_scores for m, r in run_results.items()},
        )
        with open(cfg.output_dir / "yearly_val_ic.json", "w", encoding="utf-8") as f:
            json.dump(
                {m: {str(k): v for k, v in r.yearly_val_ic.items()} for m, r in run_results.items()},
                f,
                indent=2,
            )
        if "M5_TSCSNet" in run_results and run_results["M5_TSCSNet"].gate_history:
            g_df = pd.DataFrame(
                list(run_results["M5_TSCSNet"].gate_history.items()),
                columns=["signal_date", "gate_temporal_weight"],
            )
            g_df.to_csv(cfg.output_dir / "tscsnet_gate_history.csv", index=False)

    # Equal-weight CSI 500 control under exact same engine & dates
    ew_scores = np.where(panel.valid_mask, 0.0, np.nan).astype(np.float32)
    bt_ew = run_top_k_backtest("B1_CSI500_EW", panel, ew_scores, cfg, top_k=500, hold_buffer=500)
    bt_ew_20 = run_top_k_backtest(
        "B1_CSI500_EW", panel, ew_scores, cfg, top_k=500, hold_buffer=500, flat_single_side_fee=cfg.stress_single_side_fee
    )

    bt_tilt: Dict[str, BacktestResult] = {"B1_CSI500_EW": bt_ew}
    bt_top50: Dict[str, BacktestResult] = {"B1_CSI500_EW": bt_ew}
    tilt_rows = [_b0_row(bt_ew.bench_nav)]
    top50_rows = [_b0_row(bt_ew.bench_nav)]

    ew_row = {
        "Model": "B1_CSI500_EW",
        "Params": 0,
        "RankIC": np.nan,
        "RankICIR_Ann": np.nan,
        "RankIC_WinRate": np.nan,
        "Top50_Biweekly_Excess": np.nan,
        "TopMinusBot50_Biweekly": np.nan,
        "Ann_Return": bt_ew.metrics["Ann_Return"],
        "Ann_Excess": bt_ew.metrics["Ann_Excess"],
        "IR": bt_ew.metrics["IR"],
        "Sharpe": bt_ew.metrics["Sharpe"],
        "MaxDD": bt_ew.metrics["MaxDD"],
        "Rel_MaxDD": bt_ew.metrics["Rel_MaxDD"],
        "Ann_OneWay_Turnover": bt_ew.metrics["Ann_OneWay_Turnover"],
        "Yearly_Excess_WinRate": bt_ew.metrics["Yearly_Excess_WinRate"],
        "Stress20bp_Ann_Excess": bt_ew_20.metrics["Ann_Excess"],
        "Stress20bp_IR": bt_ew_20.metrics["IR"],
        "Train_Sec": 0.0,
    }
    tilt_rows.append(ew_row)
    top50_rows.append(dict(ew_row))

    for m_name in cfg.model_names:
        res = run_results[m_name]
        ic = evaluate_signal_ic(panel, res.oos_scores, rebal_indices, top_k=cfg.top_k)

        r_tilt = run_soft_tilt_backtest(m_name, panel, res.oos_scores, cfg)
        r_tilt_20 = run_soft_tilt_backtest(
            m_name, panel, res.oos_scores, cfg, flat_single_side_fee=cfg.stress_single_side_fee
        )
        r_top = run_top_k_backtest(m_name, panel, res.oos_scores, cfg)
        r_top_20 = run_top_k_backtest(
            m_name, panel, res.oos_scores, cfg, flat_single_side_fee=cfg.stress_single_side_fee
        )
        bt_tilt[m_name] = r_tilt
        bt_top50[m_name] = r_top

        for target_list, bt_main, bt_st in (
            (tilt_rows, r_tilt, r_tilt_20),
            (top50_rows, r_top, r_top_20),
        ):
            target_list.append({
                "Model": m_name,
                "Params": res.num_params,
                "RankIC": ic["RankIC"],
                "RankICIR_Ann": ic["RankICIR_Ann"],
                "RankIC_WinRate": ic["RankIC_WinRate"],
                "Top50_Biweekly_Excess": ic["Top50_Biweekly_Excess"],
                "TopMinusBot50_Biweekly": ic["TopMinusBot50_Biweekly"],
                "Ann_Return": bt_main.metrics["Ann_Return"],
                "Ann_Excess": bt_main.metrics["Ann_Excess"],
                "IR": bt_main.metrics["IR"],
                "Sharpe": bt_main.metrics["Sharpe"],
                "MaxDD": bt_main.metrics["MaxDD"],
                "Rel_MaxDD": bt_main.metrics["Rel_MaxDD"],
                "Ann_OneWay_Turnover": bt_main.metrics["Ann_OneWay_Turnover"],
                "Yearly_Excess_WinRate": bt_main.metrics["Yearly_Excess_WinRate"],
                "Stress20bp_Ann_Excess": bt_st.metrics["Ann_Excess"],
                "Stress20bp_IR": bt_st.metrics["IR"],
                "Train_Sec": round(res.elapsed_seconds, 1),
            })

    tilt_df = pd.DataFrame(tilt_rows)
    top50_df = pd.DataFrame(top50_rows)
    tilt_df.to_csv(cfg.output_dir / "ablation_summary.csv", index=False)
    top50_df.to_csv(cfg.output_dir / "ablation_top50_summary.csv", index=False)

    # Save primary (Soft-Tilt) daily NAV, excess NAV, and yearly breakdown
    nav_df = pd.DataFrame({"date": bt_ew.dates, "B0_CSI500_TR": bt_ew.bench_nav})
    ex_df = pd.DataFrame({"date": bt_ew.dates})
    yearly_frames = []
    for k_name, bt in bt_tilt.items():
        nav_df[k_name] = bt.nav
        ex_df[k_name] = bt.excess_nav
        ydf = bt.yearly_table.copy()
        ydf.insert(0, "Model", k_name)
        yearly_frames.append(ydf)
    nav_df.to_csv(cfg.output_dir / "daily_nav.csv", index=False)
    ex_df.to_csv(cfg.output_dir / "daily_excess_nav.csv", index=False)
    pd.concat(yearly_frames, ignore_index=True).to_csv(cfg.output_dir / "yearly_breakdown.csv", index=False)

    print(f"\n=== Primary: Soft-Tilt Index Enhancement Summary ({time.time() - t_all:.1f}s) ===")
    print(tilt_df[["Model", "Params", "RankIC", "RankICIR_Ann", "RankIC_WinRate", "Ann_Return", "Ann_Excess", "IR", "Rel_MaxDD", "Ann_OneWay_Turnover", "Yearly_Excess_WinRate"]].to_string(index=False))
    print("\n=== Comparison: Concentrated Top-50 / Buffer-100 Summary ===")
    print(top50_df[["Model", "Ann_Return", "Ann_Excess", "IR", "Rel_MaxDD", "Ann_OneWay_Turnover", "Yearly_Excess_WinRate"]].to_string(index=False))


if __name__ == "__main__":
    main()
