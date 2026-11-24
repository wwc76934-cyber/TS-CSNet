"""Exact biweekly Top-K / Buffer-M open-to-open backtest and IC evaluation engine.

Key realism guarantees:
  - Signal at Friday close -> executes at next trading day (T+1) open.
  - Strict A-share tradability: no buying limit-up / suspended / ST stocks;
    no selling limit-down / suspended stocks (locked positions stay held).
  - Zero hidden leverage: transaction fees are funded strictly inside portfolio NAV.
  - Same-engine benchmarks: CSI 500 Total Return (H00905) and biweekly equal-weight
    CSI 500 (B1_CSI500_EW) under identical dates, tradability, and fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .config import ExperimentConfig
from .dataset import PanelData


@dataclass
class BacktestResult:
    name: str
    dates: np.ndarray                # Daily calendar dates within OOS window
    nav: np.ndarray                  # Daily strategy NAV (starts at 1.0)
    bench_nav: np.ndarray            # Daily H00905 Total Return NAV (starts at 1.0)
    excess_nav: np.ndarray           # Daily relative NAV = nav / bench_nav
    rebal_dates: List[str]           # Execution dates of biweekly rebalances
    rebal_turnovers: List[float]     # One-way turnover at each rebalance
    metrics: Dict[str, float]        # Summary metrics dict
    yearly_table: pd.DataFrame       # Annual return, benchmark, excess, IR, DD


def get_oos_rebalance_indices(panel: PanelData, cfg: ExperimentConfig) -> np.ndarray:
    """Return biweekly schedule indices falling strictly inside cfg.test_years."""
    years_set = {str(y) for y in cfg.test_years}
    oos_all = np.array(
        [k for k, d in enumerate(panel.signal_dates) if str(d)[:4] in years_set],
        dtype=np.int64,
    )
    return oos_all[:: cfg.rebalance_stride]


def evaluate_signal_ic(
    panel: PanelData,
    oos_scores: np.ndarray,
    rebal_indices: np.ndarray,
    top_k: int = 50,
) -> Dict[str, float]:
    """Evaluate biweekly out-of-sample Pearson IC, Spearman RankIC, ICIR, and Top-Bottom spread."""
    pearson_list: List[float] = []
    rank_list: List[float] = []
    ls_spreads: List[float] = []
    top_excess: List[float] = []

    for k in rebal_indices:
        m = panel.label_valid[k] & np.isfinite(oos_scores[k])
        if int(m.sum()) < 2 * top_k:
            continue
        s = oos_scores[k, m]
        y = panel.y_raw[k, m]
        if np.std(s) < 1e-10 or np.std(y) < 1e-10:
            continue

        p_ic = pearsonr(s, y).statistic
        r_ic = spearmanr(s, y).statistic
        if np.isfinite(p_ic):
            pearson_list.append(float(p_ic))
        if np.isfinite(r_ic):
            rank_list.append(float(r_ic))

        order = np.argsort(-s)
        top_ret = float(np.mean(y[order[:top_k]]))
        bot_ret = float(np.mean(y[order[-top_k:]]))
        all_ret = float(np.mean(y))
        ls_spreads.append(top_ret - bot_ret)
        top_excess.append(top_ret - all_ret)

    r_arr = np.array(rank_list, dtype=np.float64)
    p_arr = np.array(pearson_list, dtype=np.float64)
    ann_factor = np.sqrt(26.0)  # ~26 biweekly periods per year

    return {
        "IC_Pearson": float(p_arr.mean()) if len(p_arr) else float("nan"),
        "RankIC": float(r_arr.mean()) if len(r_arr) else float("nan"),
        "RankIC_Std": float(r_arr.std(ddof=1)) if len(r_arr) > 1 else float("nan"),
        "RankICIR_Ann": float((r_arr.mean() / (r_arr.std(ddof=1) + 1e-12)) * ann_factor) if len(r_arr) > 1 else float("nan"),
        "RankIC_WinRate": float((r_arr > 0).mean()) if len(r_arr) else float("nan"),
        "Top50_Biweekly_Excess": float(np.mean(top_excess)) if len(top_excess) else float("nan"),
        "TopMinusBot50_Biweekly": float(np.mean(ls_spreads)) if len(ls_spreads) else float("nan"),
        "Num_Periods": float(len(r_arr)),
    }


def _sell_fee_rate(date_str: str, cfg: ExperimentConfig, flat_fee: Optional[float]) -> float:
    if flat_fee is not None:
        return flat_fee
    if date_str < cfg.stamp_cut_date:
        return cfg.sell_fee_before_stamp_cut
    return cfg.sell_fee_after_stamp_cut


def _buy_fee_rate(cfg: ExperimentConfig, flat_fee: Optional[float]) -> float:
    return flat_fee if flat_fee is not None else cfg.buy_fee_rate


def _max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = nav / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min())


def run_top_k_backtest(
    name: str,
    panel: PanelData,
    oos_scores: np.ndarray,
    cfg: ExperimentConfig,
    top_k: Optional[int] = None,
    hold_buffer: Optional[int] = None,
    flat_single_side_fee: Optional[float] = None,
) -> BacktestResult:
    """Run exact daily mark-to-market biweekly Top-K / Buffer-M backtest."""
    k_target = cfg.top_k if top_k is None else top_k
    buf_rank = cfg.hold_buffer if hold_buffer is None else hold_buffer

    rebal_indices = get_oos_rebalance_indices(panel, cfg)
    rebal_map: Dict[int, int] = {int(panel.exec_cal_idx[k]): int(k) for k in rebal_indices}

    start_day = int(panel.exec_cal_idx[rebal_indices[0]])
    last_year = str(max(cfg.test_years))
    cal_str = np.asarray(panel.cal_dates, dtype="<U10")
    end_candidates = np.where(np.char.startswith(cal_str, f"{last_year}-"))[0]
    end_day = int(end_candidates[-1])

    days = np.arange(start_day, end_day + 1, dtype=np.int64)
    n_days = len(days)
    daily_nav = np.ones(n_days, dtype=np.float64)

    # Benchmark H00905 normalized at start_day - 1 close (exact alignment with start_day open entry)
    bench_base_idx = max(0, start_day - 1)
    bench_series = panel.bench_h00905[days] / panel.bench_h00905[bench_base_idx]

    pos_val: Dict[int, float] = {}
    ema_scores: Dict[int, float] = {}
    alpha_ema = float(cfg.signal_ema_alpha)
    cash = 1.0
    rebal_dates: List[str] = []
    rebal_turnovers: List[float] = []

    hfq_open = panel.hfq_open
    hfq_close = panel.hfq_close
    can_buy = panel.can_buy
    can_sell = panel.can_sell

    for i_day, d in enumerate(days):
        if d in rebal_map:
            k_sched = rebal_map[d]
            date_str = str(panel.cal_dates[d])

            # 1. Mark existing holdings from previous close to today's execution open
            if i_day > 0 and pos_val:
                for s_idx in list(pos_val.keys()):
                    prev_c = float(hfq_close[d - 1, s_idx])
                    cur_o = float(hfq_open[d, s_idx])
                    if np.isfinite(prev_c) and prev_c > 0 and np.isfinite(cur_o) and cur_o > 0:
                        pos_val[s_idx] *= (cur_o / prev_c)

            nav_open = cash + sum(pos_val.values())

            # 2. Rank active constituents on signal date k_sched (with causal EMA smoothing)
            valid_m = panel.valid_mask[k_sched] & np.isfinite(oos_scores[k_sched])
            s_codes_idx = panel.member_idx[k_sched, valid_m]
            raw_s = oos_scores[k_sched, valid_m]
            eff_s = np.empty_like(raw_s, dtype=np.float64)
            for j in range(len(s_codes_idx)):
                sid = int(s_codes_idx[j])
                val_s = float(raw_s[j])
                if sid in ema_scores and alpha_ema < 1.0:
                    sm = alpha_ema * val_s + (1.0 - alpha_ema) * ema_scores[sid]
                else:
                    sm = val_s
                ema_scores[sid] = sm
                eff_s[j] = sm

            # Deterministic tie-breaking by stock index (important for equal-weight control)
            order = np.lexsort((s_codes_idx, -eff_s))
            ranked_stocks = [int(s_codes_idx[j]) for j in order]
            rank_map = {s: r + 1 for r, s in enumerate(ranked_stocks)}

            # 3. Partition current holdings into locked (unsellable), kept (inside buffer), and to-sell
            locked: Dict[int, float] = {}
            kept: List[int] = []
            for s_idx, val in pos_val.items():
                if not bool(can_sell[d, s_idx]):
                    locked[s_idx] = val
                elif s_idx in rank_map and rank_map[s_idx] <= buf_rank:
                    kept.append(s_idx)

            # 4. Fill remaining slots up to k_target from highest-ranked buyable stocks
            selected_set = set(locked.keys()) | set(kept)
            new_buys: List[int] = []
            for s_idx in ranked_stocks:
                if len(selected_set) >= k_target:
                    break
                if s_idx in selected_set:
                    continue
                if bool(can_buy[d, s_idx]):
                    new_buys.append(s_idx)
                    selected_set.add(s_idx)

            active_stocks = kept + new_buys

            # 5. Compute target weights and execute open rebalance with exact self-financed fees
            buy_fee_r = _buy_fee_rate(cfg, flat_single_side_fee)
            sell_fee_r = _sell_fee_rate(date_str, cfg, flat_single_side_fee)

            locked_total = sum(locked.values())
            allocatable = max(nav_open - locked_total, 0.0)

            target_val: Dict[int, float] = dict(locked)
            if active_stocks and allocatable > 0:
                eq_w = allocatable / len(active_stocks)
                for s_idx in active_stocks:
                    cur_v = pos_val.get(s_idx, 0.0)
                    # Never buy extra shares of a kept stock that is currently not buyable (e.g. limit-up)
                    if eq_w > cur_v and not bool(can_buy[d, s_idx]):
                        target_val[s_idx] = cur_v
                    else:
                        target_val[s_idx] = eq_w

            # Compute sells and desired buys
            all_involved = set(pos_val.keys()) | set(target_val.keys())
            total_sell = 0.0
            desired_buy: Dict[int, float] = {}
            new_pos: Dict[int, float] = {}

            for s_idx in all_involved:
                cur_v = pos_val.get(s_idx, 0.0)
                tgt_v = target_val.get(s_idx, 0.0)
                if tgt_v < cur_v:
                    sell_amt = cur_v - tgt_v
                    total_sell += sell_amt
                    if tgt_v > 1e-12:
                        new_pos[s_idx] = tgt_v
                elif tgt_v > cur_v:
                    desired_buy[s_idx] = tgt_v - cur_v
                    if cur_v > 1e-12:
                        new_pos[s_idx] = cur_v
                else:
                    if cur_v > 1e-12:
                        new_pos[s_idx] = cur_v

            sell_proceeds = total_sell * (1.0 - sell_fee_r)
            avail_cash = cash + sell_proceeds
            total_desired_buy = sum(desired_buy.values())

            # Scale buys cleanly so cash + proceeds covers buys + buy fees with zero leverage
            if total_desired_buy > 0 and avail_cash > 0:
                max_buy = avail_cash / (1.0 + buy_fee_r)
                scale = min(1.0, max_buy / total_desired_buy)
                total_buy = 0.0
                for s_idx, b_amt in desired_buy.items():
                    exec_buy = b_amt * scale
                    new_pos[s_idx] = new_pos.get(s_idx, 0.0) + exec_buy
                    total_buy += exec_buy
                cash = max(0.0, avail_cash - total_buy * (1.0 + buy_fee_r))
            else:
                total_buy = 0.0
                cash = avail_cash

            pos_val = new_pos
            rebal_dates.append(date_str)
            rebal_turnovers.append(0.5 * (total_buy + total_sell) / max(nav_open, 1e-12))

            # 6. Mark intraday move from execution open to today's close
            for s_idx in list(pos_val.keys()):
                cur_o = float(hfq_open[d, s_idx])
                cur_c = float(hfq_close[d, s_idx])
                if np.isfinite(cur_o) and cur_o > 0 and np.isfinite(cur_c) and cur_c > 0:
                    pos_val[s_idx] *= (cur_c / cur_o)
            daily_nav[i_day] = cash + sum(pos_val.values())

        else:
            # Non-rebalance day: mark close-to-close (suspended stocks stay flat via ffill hfq_close)
            for s_idx in list(pos_val.keys()):
                prev_c = float(hfq_close[d - 1, s_idx])
                cur_c = float(hfq_close[d, s_idx])
                if np.isfinite(prev_c) and prev_c > 0 and np.isfinite(cur_c) and cur_c > 0:
                    pos_val[s_idx] *= (cur_c / prev_c)
            daily_nav[i_day] = cash + sum(pos_val.values())

    # Align daily returns (prepending 1.0 before day 0)
    dates_out = panel.cal_dates[days]
    excess_nav = daily_nav / bench_series

    nav_ext = np.concatenate([[1.0], daily_nav])
    bench_ext = np.concatenate([[1.0], bench_series])
    r_strat = nav_ext[1:] / nav_ext[:-1] - 1.0
    r_bench = bench_ext[1:] / bench_ext[:-1] - 1.0
    r_excess = r_strat - r_bench

    n_years = n_days / 242.0
    ann_ret = float(daily_nav[-1] ** (1.0 / n_years) - 1.0)
    bench_ann_ret = float(bench_series[-1] ** (1.0 / n_years) - 1.0)
    ann_excess = float(excess_nav[-1] ** (1.0 / n_years) - 1.0)

    ann_vol = float(np.std(r_strat, ddof=1) * np.sqrt(242.0))
    sharpe = ann_ret / (ann_vol + 1e-12)
    te = float(np.std(r_excess, ddof=1) * np.sqrt(242.0))
    ir = float((np.mean(r_excess) * 242.0) / (te + 1e-12))

    mdd = _max_drawdown(daily_nav)
    rel_mdd = _max_drawdown(excess_nav)
    ann_turnover = float(sum(rebal_turnovers) / n_years)

    # Yearly breakdown table
    years = np.array([int(str(d)[:4]) for d in dates_out])
    rows = []
    for y in sorted(set(years.tolist())):
        idx = np.where(years == y)[0]
        i0, i1 = idx[0], idx[-1]
        s_prev = 1.0 if i0 == 0 else daily_nav[i0 - 1]
        b_prev = 1.0 if i0 == 0 else bench_series[i0 - 1]
        y_ret = float(daily_nav[i1] / s_prev - 1.0)
        b_ret = float(bench_series[i1] / b_prev - 1.0)
        ex_curve = (daily_nav[idx] / s_prev) / (bench_series[idx] / b_prev)
        y_ex = float(ex_curve[-1] - 1.0)
        y_r_ex = r_excess[idx]
        y_ir = float((y_r_ex.mean() * 242.0) / (y_r_ex.std(ddof=1) * np.sqrt(242.0) + 1e-12))
        rows.append({
            "Year": y,
            "Strategy_Ret": y_ret,
            "CSI500_TR_Ret": b_ret,
            "Excess_Ret": y_ex,
            "IR": y_ir,
            "Rel_MaxDD": _max_drawdown(ex_curve),
        })
    yearly_df = pd.DataFrame(rows)
    yearly_win_rate = float((yearly_df["Excess_Ret"] > 0).mean())

    metrics = {
        "Ann_Return": ann_ret,
        "Bench_Ann_Return": bench_ann_ret,
        "Ann_Excess": ann_excess,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Tracking_Error": te,
        "IR": ir,
        "MaxDD": mdd,
        "Rel_MaxDD": rel_mdd,
        "Ann_OneWay_Turnover": ann_turnover,
        "Yearly_Excess_WinRate": yearly_win_rate,
        "Final_NAV": float(daily_nav[-1]),
        "Final_Excess_NAV": float(excess_nav[-1]),
    }

    return BacktestResult(
        name=name,
        dates=dates_out,
        nav=daily_nav,
        bench_nav=bench_series,
        excess_nav=excess_nav,
        rebal_dates=rebal_dates,
        rebal_turnovers=rebal_turnovers,
        metrics=metrics,
        yearly_table=yearly_df,
    )


def run_soft_tilt_backtest(
    name: str,
    panel: PanelData,
    oos_scores: np.ndarray,
    cfg: ExperimentConfig,
    gamma: float = 1.0,
    mv_scale: float = 0.6,
    cap_mult: float = 2.5,
    flat_single_side_fee: Optional[float] = None,
) -> BacktestResult:
    """Run 3-line benchmark-aligned Soft-Tilt Index Enhancement backtest.

    On each biweekly rebalance date:
      1. Causal EMA smoothing across stock scores reduces signal whipsaw.
      2. Base weight w_base_i ~ exp(mv_scale * z_log_mv_i) tracks CSI 500 float-cap structure.
      3. Tilted weight w_i ~ w_base_i * clip(1 + gamma * z_score_i, 0, cap_mult)
         overweights high-scoring stocks while zeroing out the bottom-tail losers.
    Uses the exact same A-share tradability, limit-up/down, and fee mechanics as Top-K.
    """
    rebal_indices = get_oos_rebalance_indices(panel, cfg)
    rebal_map: Dict[int, int] = {int(panel.exec_cal_idx[k]): int(k) for k in rebal_indices}

    start_day = int(panel.exec_cal_idx[rebal_indices[0]])
    last_year = str(max(cfg.test_years))
    cal_str = np.asarray(panel.cal_dates, dtype="<U10")
    end_candidates = np.where(np.char.startswith(cal_str, f"{last_year}-"))[0]
    end_day = int(end_candidates[-1])

    days = np.arange(start_day, end_day + 1, dtype=np.int64)
    n_days = len(days)
    daily_nav = np.ones(n_days, dtype=np.float64)

    bench_base_idx = max(0, start_day - 1)
    bench_series = panel.bench_h00905[days] / panel.bench_h00905[bench_base_idx]

    mv_col = cfg.static_features.index("log_float_mv")
    pos_val: Dict[int, float] = {}
    ema_scores: Dict[int, float] = {}
    alpha_ema = float(cfg.signal_ema_alpha)
    cash = 1.0
    rebal_dates: List[str] = []
    rebal_turnovers: List[float] = []

    hfq_open = panel.hfq_open
    hfq_close = panel.hfq_close
    can_buy = panel.can_buy
    can_sell = panel.can_sell

    for i_day, d in enumerate(days):
        if d in rebal_map:
            k_sched = rebal_map[d]
            date_str = str(panel.cal_dates[d])

            if i_day > 0 and pos_val:
                for s_idx in list(pos_val.keys()):
                    prev_c = float(hfq_close[d - 1, s_idx])
                    cur_o = float(hfq_open[d, s_idx])
                    if np.isfinite(prev_c) and prev_c > 0 and np.isfinite(cur_o) and cur_o > 0:
                        pos_val[s_idx] *= (cur_o / prev_c)

            nav_open = cash + sum(pos_val.values())

            valid_m = panel.valid_mask[k_sched] & np.isfinite(oos_scores[k_sched])
            sids = panel.member_idx[k_sched, valid_m]
            raw_s = oos_scores[k_sched, valid_m]
            z_mv = panel.X_stat[k_sched, valid_m, mv_col].astype(np.float64)

            eff_s = np.empty(len(sids), dtype=np.float64)
            for j, sid in enumerate(sids):
                sid_int = int(sid)
                v = float(raw_s[j])
                sm = alpha_ema * v + (1.0 - alpha_ema) * ema_scores[sid_int] if (sid_int in ema_scores and alpha_ema < 1.0) else v
                ema_scores[sid_int] = sm
                eff_s[j] = sm

            s_std = eff_s.std()
            z_s = (eff_s - eff_s.mean()) / (s_std + 1e-8) if s_std > 1e-8 else np.zeros_like(eff_s)

            w_base = np.exp(mv_scale * np.clip(z_mv, -3.0, 3.0)) if mv_scale > 0 else np.ones(len(sids))
            w_base = w_base / (w_base.sum() + 1e-12)
            mult = np.clip(1.0 + gamma * z_s, 0.0, cap_mult)
            w_tgt = w_base * mult

            # Exclude stocks that cannot be bought unless already held
            buy_or_held = np.array([bool(can_buy[d, int(sid)]) or (int(sid) in pos_val) for sid in sids])
            w_tgt = np.where(buy_or_held, w_tgt, 0.0)
            w_sum = w_tgt.sum()
            if w_sum > 1e-12:
                w_tgt = w_tgt / w_sum

            locked = {s_idx: val for s_idx, val in pos_val.items() if not bool(can_sell[d, s_idx])}
            allocatable = max(nav_open - sum(locked.values()), 0.0)

            target_val: Dict[int, float] = dict(locked)
            for j, sid in enumerate(sids):
                sid_int = int(sid)
                if sid_int in locked:
                    continue
                tv = allocatable * float(w_tgt[j])
                cv = pos_val.get(sid_int, 0.0)
                if tv > cv and not bool(can_buy[d, sid_int]):
                    tv = cv
                if tv > 1e-12:
                    target_val[sid_int] = tv

            buy_fee_r = _buy_fee_rate(cfg, flat_single_side_fee)
            sell_fee_r = _sell_fee_rate(date_str, cfg, flat_single_side_fee)

            total_sell = 0.0
            desired_buy: Dict[int, float] = {}
            new_pos: Dict[int, float] = {}
            for s_idx in set(pos_val.keys()) | set(target_val.keys()):
                cv = pos_val.get(s_idx, 0.0)
                tv = target_val.get(s_idx, 0.0)
                if tv < cv:
                    total_sell += (cv - tv)
                    if tv > 1e-12:
                        new_pos[s_idx] = tv
                elif tv > cv:
                    desired_buy[s_idx] = tv - cv
                    if cv > 1e-12:
                        new_pos[s_idx] = cv
                elif cv > 1e-12:
                    new_pos[s_idx] = cv

            avail_cash = cash + total_sell * (1.0 - sell_fee_r)
            total_desired_buy = sum(desired_buy.values())
            if total_desired_buy > 0 and avail_cash > 0:
                scale = min(1.0, (avail_cash / (1.0 + buy_fee_r)) / total_desired_buy)
                total_buy = 0.0
                for s_idx, b_amt in desired_buy.items():
                    eb = b_amt * scale
                    new_pos[s_idx] = new_pos.get(s_idx, 0.0) + eb
                    total_buy += eb
                cash = max(0.0, avail_cash - total_buy * (1.0 + buy_fee_r))
            else:
                total_buy = 0.0
                cash = avail_cash

            pos_val = new_pos
            rebal_dates.append(date_str)
            rebal_turnovers.append(0.5 * (total_buy + total_sell) / max(nav_open, 1e-12))

            for s_idx in list(pos_val.keys()):
                cur_o = float(hfq_open[d, s_idx])
                cur_c = float(hfq_close[d, s_idx])
                if np.isfinite(cur_o) and cur_o > 0 and np.isfinite(cur_c) and cur_c > 0:
                    pos_val[s_idx] *= (cur_c / cur_o)
            daily_nav[i_day] = cash + sum(pos_val.values())
        else:
            for s_idx in list(pos_val.keys()):
                prev_c = float(hfq_close[d - 1, s_idx])
                cur_c = float(hfq_close[d, s_idx])
                if np.isfinite(prev_c) and prev_c > 0 and np.isfinite(cur_c) and cur_c > 0:
                    pos_val[s_idx] *= (cur_c / prev_c)
            daily_nav[i_day] = cash + sum(pos_val.values())

    dates_out = panel.cal_dates[days]
    excess_nav = daily_nav / bench_series
    nav_ext = np.concatenate([[1.0], daily_nav])
    bench_ext = np.concatenate([[1.0], bench_series])
    r_strat = nav_ext[1:] / nav_ext[:-1] - 1.0
    r_bench = bench_ext[1:] / bench_ext[:-1] - 1.0
    r_excess = r_strat - r_bench

    n_years = n_days / 242.0
    ann_ret = float(daily_nav[-1] ** (1.0 / n_years) - 1.0)
    bench_ann_ret = float(bench_series[-1] ** (1.0 / n_years) - 1.0)
    ann_excess = float(excess_nav[-1] ** (1.0 / n_years) - 1.0)
    ann_vol = float(np.std(r_strat, ddof=1) * np.sqrt(242.0))
    sharpe = ann_ret / (ann_vol + 1e-12)
    te = float(np.std(r_excess, ddof=1) * np.sqrt(242.0))
    ir = float((np.mean(r_excess) * 242.0) / (te + 1e-12))
    mdd = _max_drawdown(daily_nav)
    rel_mdd = _max_drawdown(excess_nav)
    ann_turnover = float(sum(rebal_turnovers) / n_years)

    years = np.array([int(str(d)[:4]) for d in dates_out])
    rows = []
    for y in sorted(set(years.tolist())):
        idx = np.where(years == y)[0]
        i0, i1 = idx[0], idx[-1]
        s_prev = 1.0 if i0 == 0 else daily_nav[i0 - 1]
        b_prev = 1.0 if i0 == 0 else bench_series[i0 - 1]
        y_ret = float(daily_nav[i1] / s_prev - 1.0)
        b_ret = float(bench_series[i1] / b_prev - 1.0)
        ex_curve = (daily_nav[idx] / s_prev) / (bench_series[idx] / b_prev)
        y_ex = float(ex_curve[-1] - 1.0)
        y_r_ex = r_excess[idx]
        y_ir = float((y_r_ex.mean() * 242.0) / (y_r_ex.std(ddof=1) * np.sqrt(242.0) + 1e-12))
        rows.append({
            "Year": y,
            "Strategy_Ret": y_ret,
            "CSI500_TR_Ret": b_ret,
            "Excess_Ret": y_ex,
            "IR": y_ir,
            "Rel_MaxDD": _max_drawdown(ex_curve),
        })
    yearly_df = pd.DataFrame(rows)

    metrics = {
        "Ann_Return": ann_ret,
        "Bench_Ann_Return": bench_ann_ret,
        "Ann_Excess": ann_excess,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Tracking_Error": te,
        "IR": ir,
        "MaxDD": mdd,
        "Rel_MaxDD": rel_mdd,
        "Ann_OneWay_Turnover": ann_turnover,
        "Yearly_Excess_WinRate": float((yearly_df["Excess_Ret"] > 0).mean()),
        "Final_NAV": float(daily_nav[-1]),
        "Final_Excess_NAV": float(excess_nav[-1]),
    }

    return BacktestResult(
        name=name,
        dates=dates_out,
        nav=daily_nav,
        bench_nav=bench_series,
        excess_nav=excess_nav,
        rebal_dates=rebal_dates,
        rebal_turnovers=rebal_turnovers,
        metrics=metrics,
        yearly_table=yearly_df,
    )
