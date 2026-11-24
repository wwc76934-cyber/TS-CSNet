"""Step 4: Deep Holding Analysis, Attribution Analysis (Model/Decile/Style/Brinson),
and Applicability Analysis (Regime/Style/Capacity/Cost) for the Midterm Report.

All calculations run strictly offline on saved 2017-2025 OOS predictions and
local PIT Shenwan L1 industry classifications.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tscs_alpha.backtest import get_oos_rebalance_indices, run_soft_tilt_backtest
from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import PanelData, load_panel

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

INDUSTRY_PATH = Path(r"D:\QuantProjects\dfl-csi-alpha\data\incremental\industry_map.parquet")


def load_pit_industry_lookup(panel: PanelData) -> Dict[str, List[Tuple[str, str]]]:
    """Load Shenwan L1 industry map sorted by start_date for strict PIT lookup."""
    if not INDUSTRY_PATH.exists():
        return {}
    df = pd.read_parquet(INDUSTRY_PATH).copy()
    df["start_str"] = pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values(["code", "start_str"])
    lookup: Dict[str, List[Tuple[str, str]]] = {}
    for code, grp in df.groupby("code"):
        entries = []
        for s_dt, ind in zip(grp["start_str"], grp["industry"]):
            # Clean label like 'SW_801080_电子' -> '电子'
            clean_ind = str(ind).split("_")[-1] if "_" in str(ind) else str(ind)
            entries.append((str(s_dt), clean_ind))
        lookup[str(code)] = entries
    return lookup


def get_industry_at(lookup: Dict[str, List[Tuple[str, str]]], code: str, date_str: str) -> str:
    entries = lookup.get(code)
    if not entries:
        return "其他/未知"
    res = "其他/未知"
    for s_dt, ind in entries:
        if s_dt <= date_str:
            res = ind
        else:
            break
    return res


def _compute_period_weights(
    panel: PanelData,
    scores: np.ndarray,
    rebal_indices: np.ndarray,
    cfg: ExperimentConfig,
    gamma: float = 1.0,
    mv_scale: float = 0.6,
    cap_mult: float = 2.5,
) -> Tuple[List[Dict[int, float]], List[Dict[int, float]], List[float]]:
    """Reconstruct exact biweekly portfolio weights (strategy & float-cap base) for attribution."""
    mv_col = cfg.static_features.index("log_float_mv")
    alpha_ema = float(cfg.signal_ema_alpha)
    ema: Dict[int, float] = {}

    strat_w_list: List[Dict[int, float]] = []
    bench_w_list: List[Dict[int, float]] = []
    hhi_list: List[float] = []

    for k in rebal_indices:
        vm = panel.valid_mask[k] & np.isfinite(scores[k])
        sids = panel.member_idx[k, vm]
        raw_s = scores[k, vm]
        z_mv = panel.X_stat[k, vm, mv_col].astype(np.float64)

        eff = np.empty(len(sids), dtype=np.float64)
        for j, sid in enumerate(sids):
            sid_int = int(sid)
            v = float(raw_s[j])
            sm = alpha_ema * v + (1.0 - alpha_ema) * ema[sid_int] if (sid_int in ema and alpha_ema < 1.0) else v
            ema[sid_int] = sm
            eff[j] = sm

        z_s = (eff - eff.mean()) / (eff.std() + 1e-8)
        w_base = np.exp(mv_scale * np.clip(z_mv, -3.0, 3.0))
        w_base = w_base / (w_base.sum() + 1e-12)
        mult = np.clip(1.0 + gamma * z_s, 0.0, cap_mult)
        w_tgt = w_base * mult
        w_tgt = w_tgt / (w_tgt.sum() + 1e-12)

        sw = {int(sids[j]): float(w_tgt[j]) for j in range(len(sids)) if w_tgt[j] > 1e-9}
        bw = {int(sids[j]): float(w_base[j]) for j in range(len(sids)) if w_base[j] > 1e-9}
        strat_w_list.append(sw)
        bench_w_list.append(bw)
        hhi_list.append(float(np.sum(np.array(list(sw.values())) ** 2)))

    return strat_w_list, bench_w_list, hhi_list


def run_brinson_and_holding_analysis(
    panel: PanelData,
    m5_scores: np.ndarray,
    rebal_indices: np.ndarray,
    cfg: ExperimentConfig,
    ind_lookup: Dict[str, List[Tuple[str, str]]],
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute holding profile, Brinson industry attribution, style exposure, and decile attribution."""
    strat_w_list, bench_w_list, hhi_list = _compute_period_weights(panel, m5_scores, rebal_indices, cfg)

    # 1. Holding Analysis Statistics
    n_held_list = [len(sw) for sw in strat_w_list]
    n_zeroed_list = [len(bw) - len(sw) for sw, bw in zip(strat_w_list, bench_w_list)]
    top10_w_list = [sum(sorted(sw.values(), reverse=True)[:10]) for sw in strat_w_list]
    max1_w_list = [max(sw.values()) for sw in strat_w_list]
    eff_n_list = [1.0 / h for h in hhi_list]
    active_share_list = [
        0.5 * sum(abs(sw.get(s, 0.0) - bw.get(s, 0.0)) for s in set(sw) | set(bw))
        for sw, bw in zip(strat_w_list, bench_w_list)
    ]

    holding_stats = {
        "Avg_Num_Holdings": float(np.mean(n_held_list)),
        "Min_Num_Holdings": float(np.min(n_held_list)),
        "Max_Num_Holdings": float(np.max(n_held_list)),
        "Avg_Zeroed_Tail_Stocks": float(np.mean(n_zeroed_list)),
        "Avg_Top10_Weight": float(np.mean(top10_w_list)),
        "Avg_Max_Single_Weight": float(np.mean(max1_w_list)),
        "Avg_Effective_N": float(np.mean(eff_n_list)),
        "Avg_Active_Share": float(np.mean(active_share_list)),
    }

    # 2. Brinson-Fachler Industry Attribution (Period-by-Period, then annualized & by industry)
    brinson_ind_alloc: Dict[str, List[float]] = {}
    brinson_ind_select: Dict[str, List[float]] = {}
    brinson_ind_inter: Dict[str, List[float]] = {}
    brinson_ind_active_w: Dict[str, List[float]] = {}
    brinson_ind_strat_w: Dict[str, List[float]] = {}

    yearly_brinson: Dict[int, Dict[str, float]] = {}

    # 3. Style Exposure & Style Factor Return Attribution
    style_names = list(cfg.static_features) + ["mean_ret20", "mean_turn20"]
    style_exp_records: Dict[str, List[float]] = {sn: [] for sn in style_names}
    yearly_style_exp: Dict[int, Dict[str, List[float]]] = {}

    # 4. Long (Overweight) vs Short-Avoidance (Underweight / Zeroed) Attribution
    overweight_contrib_list: List[float] = []
    underweight_contrib_list: List[float] = []
    zeroed_contrib_list: List[float] = []

    for idx_pos, k in enumerate(rebal_indices):
        if not panel.label_valid[k].any():
            continue
        date_str = str(panel.signal_dates[k])
        year = int(date_str[:4])
        sw = strat_w_list[idx_pos]
        bw = bench_w_list[idx_pos]

        # Build stock -> return & features map on date k
        vm = panel.label_valid[k]
        sids = panel.member_idx[k, vm]
        rets = panel.y_raw[k, vm].astype(np.float64)
        if len(sids) < 100:
            continue

        ret_map = {int(sids[j]): float(rets[j]) for j in range(len(sids))}
        # Re-normalize weights over labeled stocks for exact Brinson identity
        sw_l = {s: w for s, w in sw.items() if s in ret_map}
        bw_l = {s: w for s, w in bw.items() if s in ret_map}
        sw_sum = sum(sw_l.values()) + 1e-12
        bw_sum = sum(bw_l.values()) + 1e-12
        sw_l = {s: w / sw_sum for s, w in sw_l.items()}
        bw_l = {s: w / bw_sum for s, w in bw_l.items()}

        r_bench_tot = sum(bw_l[s] * ret_map[s] for s in bw_l)

        # Overweight vs Underweight/Zeroed decomposition of exact active return
        ow_c, uw_c, zr_c = 0.0, 0.0, 0.0
        for s in set(sw_l) | set(bw_l):
            dw = sw_l.get(s, 0.0) - bw_l.get(s, 0.0)
            r_rel = ret_map[s] - r_bench_tot
            if dw > 0:
                ow_c += dw * r_rel
            else:
                uw_c += dw * r_rel
                if sw_l.get(s, 0.0) == 0.0:
                    zr_c += dw * r_rel
        overweight_contrib_list.append(ow_c)
        underweight_contrib_list.append(uw_c)
        zeroed_contrib_list.append(zr_c)

        # Brinson-Fachler per Shenwan L1 industry
        ind_stocks: Dict[str, List[int]] = {}
        for s in set(sw_l) | set(bw_l):
            ind = get_industry_at(ind_lookup, str(panel.stock_codes[s]), date_str)
            ind_stocks.setdefault(ind, []).append(s)

        yb = yearly_brinson.setdefault(year, {"Allocation": 0.0, "Selection": 0.0, "Interaction": 0.0, "Total_Gross_Active": 0.0})
        for ind, slist in ind_stocks.items():
            wp = sum(sw_l.get(s, 0.0) for s in slist)
            wb = sum(bw_l.get(s, 0.0) for s in slist)
            rp = sum(sw_l.get(s, 0.0) * ret_map[s] for s in slist) / wp if wp > 1e-12 else r_bench_tot
            rb = sum(bw_l.get(s, 0.0) * ret_map[s] for s in slist) / wb if wb > 1e-12 else r_bench_tot

            alloc = (wp - wb) * (rb - r_bench_tot)
            select = wb * (rp - rb)
            inter = (wp - wb) * (rp - rb)

            brinson_ind_alloc.setdefault(ind, []).append(alloc)
            brinson_ind_select.setdefault(ind, []).append(select)
            brinson_ind_inter.setdefault(ind, []).append(inter)
            brinson_ind_active_w.setdefault(ind, []).append(wp - wb)
            brinson_ind_strat_w.setdefault(ind, []).append(wp)

            yb["Allocation"] += alloc
            yb["Selection"] += select
            yb["Interaction"] += inter
            yb["Total_Gross_Active"] += (alloc + select + inter)

        # Active Style Exposure (portfolio weighted avg z-score minus benchmark weighted avg z-score)
        stat_mat = panel.X_stat[k, vm, :].astype(np.float64)
        m_ret20 = panel.X_seq[k, vm, :, 0].mean(axis=1).astype(np.float64)
        m_turn20 = panel.X_seq[k, vm, :, 4].mean(axis=1).astype(np.float64)
        feat_mat = np.column_stack([stat_mat, m_ret20, m_turn20])
        sid_to_row = {int(sids[j]): j for j in range(len(sids))}

        w_active_vec = np.array([sw_l.get(int(s), 0.0) - bw_l.get(int(s), 0.0) for s in sids], dtype=np.float64)
        ys = yearly_style_exp.setdefault(year, {sn: [] for sn in style_names})
        for fi, sn in enumerate(style_names):
            exp_val = float(np.dot(w_active_vec, feat_mat[:, fi]))
            style_exp_records[sn].append(exp_val)
            ys[sn].append(exp_val)

    # Summarize Brinson by industry (annualized across 9 years)
    n_years = len(cfg.test_years)
    ind_rows = []
    for ind in sorted(brinson_ind_strat_w.keys()):
        tot_alloc = sum(brinson_ind_alloc[ind]) / n_years
        tot_sel = sum(brinson_ind_select[ind]) / n_years
        tot_int = sum(brinson_ind_inter[ind]) / n_years
        ind_rows.append({
            "Industry": ind,
            "Avg_Strat_Weight": float(np.mean(brinson_ind_strat_w[ind])),
            "Avg_Active_Weight": float(np.mean(brinson_ind_active_w[ind])),
            "Ann_Allocation": tot_alloc,
            "Ann_Selection": tot_sel,
            "Ann_Interaction": tot_int,
            "Ann_Total_Contrib": tot_alloc + tot_sel + tot_int,
        })
    ind_df = pd.DataFrame(ind_rows).sort_values("Ann_Total_Contrib", ascending=False).reset_index(drop=True)

    yearly_brinson_df = pd.DataFrame([
        {"Year": y, **vals} for y, vals in sorted(yearly_brinson.items())
    ])

    style_rows = []
    for sn in style_names:
        row = {"Factor": sn, "Overall_Active_Z": float(np.mean(style_exp_records[sn]))}
        for y in cfg.test_years:
            row[f"Y{y}"] = float(np.mean(yearly_style_exp[y][sn])) if y in yearly_style_exp else np.nan
        style_rows.append(row)
    style_df = pd.DataFrame(style_rows)

    # Append long-vs-short-avoidance decomposition into holding_stats
    holding_stats["Ann_Gross_Overweight_Contrib"] = float(sum(overweight_contrib_list) / n_years)
    holding_stats["Ann_Gross_Underweight_Contrib"] = float(sum(underweight_contrib_list) / n_years)
    holding_stats["Ann_Gross_Zeroed_Tail_Contrib"] = float(sum(zeroed_contrib_list) / n_years)
    holding_stats["Ann_Brinson_Allocation"] = float(yearly_brinson_df["Allocation"].sum() / n_years)
    holding_stats["Ann_Brinson_Selection"] = float(yearly_brinson_df["Selection"].sum() / n_years)
    holding_stats["Ann_Brinson_Interaction"] = float(yearly_brinson_df["Interaction"].sum() / n_years)

    return holding_stats, ind_df, yearly_brinson_df, style_df


def run_applicability_analysis(
    panel: PanelData,
    m5_scores: np.ndarray,
    rebal_indices: np.ndarray,
    cfg: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate strategy applicability across market regimes, volatility/beta regimes, and capital scales."""
    nav_df = pd.read_csv(cfg.output_dir / "daily_nav.csv")
    ex_df = pd.read_csv(cfg.output_dir / "daily_excess_nav.csv")
    dates = pd.to_datetime(nav_df["date"])
    b0 = nav_df["B0_CSI500_TR"].to_numpy()
    m5 = nav_df["M5_TSCSNet"].to_numpy()

    r_b0 = np.diff(np.concatenate([[1.0], b0])) / np.concatenate([[1.0], b0[:-1]])
    r_m5 = np.diff(np.concatenate([[1.0], m5])) / np.concatenate([[1.0], m5[:-1]])
    r_ex = r_m5 - r_b0

    # Portfolio daily Beta & Alpha (CAPM regression on H00905)
    cov_mb = np.cov(r_m5, r_b0, ddof=1)[0, 1]
    var_b = np.var(r_b0, ddof=1)
    port_beta = float(cov_mb / (var_b + 1e-12))
    capm_alpha_ann = float((np.mean(r_m5) - port_beta * np.mean(r_b0)) * 242.0)

    # 1. Market Regime Split (based on rolling 60-day CSI 500 trend: Bull > +8%, Bear < -8%, Sideways [-8%, +8%])
    b0_s = pd.Series(b0)
    ret_60d = (b0_s / b0_s.shift(60) - 1.0).fillna(0.0).to_numpy()
    vol_20d = pd.Series(r_b0).rolling(20, min_periods=5).std().bfill().to_numpy() * np.sqrt(242.0)
    vol_med = float(np.median(vol_20d))

    regimes = [
        ("单边上涨牛市 (60日涨幅 > +8%)", ret_60d > 0.08),
        ("震荡整理市 (-8% ≤ 60日涨幅 ≤ +8%)", (ret_60d >= -0.08) & (ret_60d <= 0.08)),
        ("下跌熊市 (60日跌幅 < -8%)", ret_60d < -0.08),
        ("高波动行情 (20日年化波动 > 中位数)", vol_20d > vol_med),
        ("低波动行情 (20日年化波动 ≤ 中位数)", vol_20d <= vol_med),
    ]

    regime_rows = []
    for label, mask in regimes:
        n_d = int(mask.sum())
        if n_d < 20:
            continue
        rb = r_b0[mask]
        rm = r_m5[mask]
        re = r_ex[mask]
        ann_b = float(np.mean(rb) * 242.0)
        ann_m = float(np.mean(rm) * 242.0)
        ann_e = float(np.mean(re) * 242.0)
        ir = float(ann_e / (np.std(re, ddof=1) * np.sqrt(242.0) + 1e-12))
        win_d = float((re > 0).mean())
        beta_r = float(np.cov(rm, rb, ddof=1)[0, 1] / (np.var(rb, ddof=1) + 1e-12))
        regime_rows.append({
            "Regime": label,
            "Days": n_d,
            "Day_Share": n_d / len(r_b0),
            "Bench_Ann_Ret": ann_b,
            "Strat_Ann_Ret": ann_m,
            "Ann_Excess_Arith": ann_e,
            "IR": ir,
            "Daily_Excess_WinRate": win_d,
            "Portfolio_Beta": beta_r,
        })
    regime_df = pd.DataFrame(regime_rows)

    # 2. Historical Macro Cycle Split (4 distinct A-share eras)
    yr_arr = dates.dt.year.to_numpy()
    eras = [
        ("2017–2018 (蓝筹白马与去杠杆熊市)", (yr_arr >= 2017) & (yr_arr <= 2018)),
        ("2019–2021 (核心资产与赛道成长牛)", (yr_arr >= 2019) & (yr_arr <= 2021)),
        ("2022–2024 (熊市震荡与微盘波动期)", (yr_arr >= 2022) & (yr_arr <= 2024)),
        ("2025 (科技成长高Beta单边牛市)", yr_arr == 2025),
    ]
    era_rows = []
    for label, mask in eras:
        idx = np.where(mask)[0]
        i0, i1 = idx[0], idx[-1]
        ny = len(idx) / 242.0
        sp = 1.0 if i0 == 0 else m5[i0 - 1]
        bp = 1.0 if i0 == 0 else b0[i0 - 1]
        s_ret = float((m5[i1] / sp) ** (1.0 / ny) - 1.0)
        b_ret = float((b0[i1] / bp) ** (1.0 / ny) - 1.0)
        ex_c = (m5[idx] / sp) / (b0[idx] / bp)
        ex_ann = float(ex_c[-1] ** (1.0 / ny) - 1.0)
        re = r_ex[idx]
        ir = float((re.mean() * 242.0) / (re.std(ddof=1) * np.sqrt(242.0) + 1e-12))
        rmdd = float((ex_c / np.maximum.accumulate(ex_c) - 1.0).min())

        # Mean RankIC in this era
        era_ics = []
        for k in rebal_indices:
            y_k = int(str(panel.signal_dates[k])[:4])
            if (label.startswith("2017") and 2017 <= y_k <= 2018) or \
               (label.startswith("2019") and 2019 <= y_k <= 2021) or \
               (label.startswith("2022") and 2022 <= y_k <= 2024) or \
               (label.startswith("2025") and y_k == 2025):
                vm = panel.label_valid[k] & np.isfinite(m5_scores[k])
                if vm.sum() >= 50:
                    r = spearmanr(m5_scores[k, vm], panel.y_raw[k, vm]).statistic
                    if np.isfinite(r):
                        era_ics.append(float(r))
        era_rows.append({
            "Era": label,
            "Mean_RankIC": float(np.mean(era_ics)) if era_ics else np.nan,
            "Bench_Ann_Ret": b_ret,
            "Strat_Ann_Ret": s_ret,
            "Ann_Excess": ex_ann,
            "IR": ir,
            "Rel_MaxDD": rmdd,
        })
    era_df = pd.DataFrame(era_rows)

    # 3. Cost & Execution Sensitivity (Realistic vs 10bp / 15bp / 20bp / 30bp single-side)
    cost_rows = []
    for cname, fee_val in [
        ("零摩擦理论上限 (0bp)", 0.0),
        ("低成本机构费率 (10bp 单边)", 0.0010),
        ("A股分段真实成本 (买10bp/卖15~20bp)", None),
        ("偏高冲击成本压力 (20bp 单边)", 0.0020),
        ("极端高摩擦压力 (30bp 单边)", 0.0030),
    ]:
        bt = run_soft_tilt_backtest("M5", panel, m5_scores, cfg, flat_single_side_fee=fee_val)
        mt = bt.metrics
        cost_rows.append({
            "Cost_Scenario": cname,
            "Ann_Return": mt["Ann_Return"],
            "Ann_Excess": mt["Ann_Excess"],
            "IR": mt["IR"],
            "Rel_MaxDD": mt["Rel_MaxDD"],
            "Ann_Turnover": mt["Ann_OneWay_Turnover"],
            "CAPM_Portfolio_Beta": port_beta,
            "CAPM_Ann_Alpha_Realistic": capm_alpha_ann,
        })
    cost_df = pd.DataFrame(cost_rows)
    return regime_df, era_df, cost_df


def make_midterm_extra_figures(
    ind_df: pd.DataFrame,
    yearly_brinson_df: pd.DataFrame,
    style_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    era_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate 3 dedicated midterm report charts (Brinson/Style, Regime Applicability, Cost Sensitivity)."""
    # Figure 5: Brinson Industry Attribution + Active Style Exposure
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Left: Yearly Brinson Decomposition (Allocation vs Selection + Interaction)
    ax = axes[0]
    yrs = yearly_brinson_df["Year"].to_numpy()
    x = np.arange(len(yrs))
    alloc = yearly_brinson_df["Allocation"].to_numpy() * 100.0
    sel_int = (yearly_brinson_df["Selection"] + yearly_brinson_df["Interaction"]).to_numpy() * 100.0
    w = 0.36
    ax.bar(x - w / 2, alloc, width=w, color="#4e79a7", label="行业配置效应 (Allocation)")
    ax.bar(x + w / 2, sel_int, width=w, color="#1f3b73", label="行业内个股选择效应 (Selection+Inter)")
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in yrs], fontsize=9.5)
    ax.set_ylabel("毛超额收益贡献 (%)", fontsize=10.5)
    ax.set_title("Brinson 业绩归因：行业配置 vs 行业内个股选择", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8.8, frameon=True)

    # Right: Top/Bottom Shenwan L1 Industries by Annualized Contribution + Active Style Tilt inset
    ax2 = axes[1]
    top_bot = pd.concat([ind_df.head(6), ind_df.tail(5)]).sort_values("Ann_Total_Contrib")
    y_pos = np.arange(len(top_bot))
    vals = top_bot["Ann_Total_Contrib"].to_numpy() * 100.0
    cols = ["#1f3b73" if v >= 0 else "#d9534f" for v in vals]
    ax2.barh(y_pos, vals, color=cols, height=0.62)
    ax2.axvline(0, color="#333", linewidth=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_bot["Industry"], fontsize=9.5)
    ax2.set_xlabel("年化超额收益贡献 (%)", fontsize=10.5)
    ax2.set_title("申万一级行业年化超额贡献 Top 6 与 Bottom 5", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.set_xlim(float(vals.min()) - 0.22, float(vals.max()) + 0.22)
    for yp, v in zip(y_pos, vals):
        ax2.text(v + (0.02 if v >= 0 else -0.02), yp, f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "05_brinson_industry_attribution.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 6: Applicability Analysis (Market Regime + Historical Eras + Active Style Tilt)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # Left: Market Regime Excess & IR
    ax = axes[0]
    short_reg = ["单边牛市\n(> +8%)", "震荡市\n(-8%~+8%)", "下跌熊市\n(< -8%)", "高波动\n行情", "低波动\n行情"]
    vals_r = regime_df["Ann_Excess_Arith"].to_numpy() * 100.0
    irs_r = regime_df["IR"].to_numpy()
    cols_r = ["#1f3b73" if v >= 0 else "#d9534f" for v in vals_r]
    xr = np.arange(len(vals_r))
    bars = ax.bar(xr, vals_r, color=cols_r, width=0.56)
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xticks(xr)
    ax.set_xticklabels(short_reg[: len(vals_r)], fontsize=9.2)
    ax.set_ylabel("分阶段年化超额收益 (%)", fontsize=10.5)
    ax.set_title("适用性分析（一）：不同行情与波动环境超额表现", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    span = max(3.0, float(np.max(np.abs(vals_r))))
    ax.set_ylim(float(np.min(vals_r)) - 0.30 * span, float(np.max(vals_r)) + 0.30 * span)
    for b, v, ir in zip(bars, vals_r, irs_r):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            v + (0.04 * span if v >= 0 else -0.05 * span),
            f"{v:+.1f}%\n(IR {ir:+.2f})",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8.5,
        )

    # Right: Active Style Exposure Z-Score of M5_TSCSNet
    ax2 = axes[1]
    factor_labels = {
        "ep_ttm": "盈利收益率 (EP)",
        "bp_mrq": "账面市值比 (BP)",
        "roe_ttm": "隐含 ROE",
        "roe_chg_60d": "60日 ROE 改善",
        "log_float_mv": "流通市值 (Size)",
        "mom_60d": "60日动量",
        "vol_60d": "60日波动率",
        "turn_cv_20d": "换手率变异系数",
        "mean_ret20": "20日短期收益",
        "mean_turn20": "20日平均换手率",
    }
    s_sorted = style_df.copy()
    s_sorted["Label"] = s_sorted["Factor"].map(lambda x: factor_labels.get(x, x))
    s_sorted = s_sorted.sort_values("Overall_Active_Z")
    yp = np.arange(len(s_sorted))
    zv = s_sorted["Overall_Active_Z"].to_numpy()
    zc = ["#2e7d32" if v >= 0 else "#c62828" for v in zv]
    ax2.barh(yp, zv, color=zc, height=0.62)
    ax2.axvline(0, color="#333", linewidth=0.9)
    ax2.set_yticks(yp)
    ax2.set_yticklabels(s_sorted["Label"], fontsize=9.2)
    ax2.set_xlabel("相对基准主动暴露 (截面 Z-Score)", fontsize=10.5)
    ax2.set_title("风格归因：M5 TS-CSNet 相对基准的平均主动因子暴露", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.set_xlim(float(zv.min()) - 0.14, float(zv.max()) + 0.12)
    for y_i, v in zip(yp, zv):
        ax2.text(v + (0.015 if v >= 0 else -0.015), y_i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_dir / "06_regime_and_style_applicability.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 7: Cost Sensitivity & Historical Era Comparison
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))

    ax = axes[0]
    era_labels = ["2017–2018\n蓝筹/去杠杆", "2019–2021\n核心资产牛", "2022–2024\n熊市/微盘震荡", "2025\n高Beta科技牛"]
    xe = np.arange(len(era_df))
    ex_v = era_df["Ann_Excess"].to_numpy() * 100.0
    ic_v = era_df["Mean_RankIC"].to_numpy() * 100.0
    bars_e = ax.bar(xe, ex_v, color=["#1f3b73" if v >= 0 else "#d9534f" for v in ex_v], width=0.52, label="年化净超额 (%)")
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xticks(xe)
    ax.set_xticklabels(era_labels, fontsize=9.2)
    ax.set_ylabel("年化净超额收益 (%)", fontsize=10.2)
    ax.set_title("适用性分析（二）：分宏观周期的 RankIC 与净超额对比", fontsize=11.2, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax_r = ax.twinx()
    ax_r.plot(xe, ic_v, color="#f28e2b", marker="o", linewidth=2.2, markersize=7, label="期均 RankIC (%)")
    ax_r.set_ylabel("期均双周 RankIC (%)", fontsize=10.2, color="#d97706")
    ax_r.set_ylim(0, max(12.0, float(np.nanmax(ic_v)) + 3.0))
    for i_x, v_ic in zip(xe, ic_v):
        ax_r.text(
            i_x,
            v_ic + 0.55,
            f"IC {v_ic:+.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#b45309",
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.88, edgecolor="#f28e2b", linewidth=0.6, boxstyle="round,pad=0.2"),
        )

    ax2 = axes[1]
    c_labels = ["0bp\n理论上限", "10bp\n低费机构", "A股分段\n真实成本", "20bp\n较高冲击", "30bp\n极端摩擦"]
    xc = np.arange(len(cost_df))
    cv = cost_df["Ann_Excess"].to_numpy() * 100.0
    bars_c = ax2.bar(xc, cv, color="#1f3b73", width=0.55)
    ax2.axhline(0, color="#333", linewidth=0.9)
    ax2.set_xticks(xc)
    ax2.set_xticklabels(c_labels, fontsize=9.2)
    ax2.set_ylabel("年化净超额收益 (%)", fontsize=10.2)
    ax2.set_title("成本稳健性：不同单边交易摩擦下的净超额衰减", fontsize=11.2, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.set_ylim(min(0.0, float(cv.min()) - 0.5), float(cv.max()) + 0.7)
    for b, v in zip(bars_c, cv):
        ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.08, f"{v:+.2f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_dir / "07_era_and_cost_applicability.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = ExperimentConfig()
    panel = load_panel(cfg)
    rebal_indices = get_oos_rebalance_indices(panel, cfg)
    scores_npz = np.load(cfg.output_dir / "oos_scores.npz")
    m5_scores = scores_npz["M5_TSCSNet"]

    ind_lookup = load_pit_industry_lookup(panel)
    holding_stats, ind_df, yearly_brinson_df, style_df = run_brinson_and_holding_analysis(
        panel, m5_scores, rebal_indices, cfg, ind_lookup
    )
    regime_df, era_df, cost_df = run_applicability_analysis(panel, m5_scores, rebal_indices, cfg)

    # Save all analysis tables
    with open(cfg.output_dir / "holding_and_attribution_summary.json", "w", encoding="utf-8") as f:
        json.dump(holding_stats, f, indent=2, ensure_ascii=False)
    ind_df.to_csv(cfg.output_dir / "brinson_industry_table.csv", index=False)
    yearly_brinson_df.to_csv(cfg.output_dir / "brinson_yearly_table.csv", index=False)
    style_df.to_csv(cfg.output_dir / "style_exposure_table.csv", index=False)
    regime_df.to_csv(cfg.output_dir / "regime_applicability.csv", index=False)
    era_df.to_csv(cfg.output_dir / "era_applicability.csv", index=False)
    cost_df.to_csv(cfg.output_dir / "cost_applicability.csv", index=False)

    make_midterm_extra_figures(ind_df, yearly_brinson_df, style_df, regime_df, era_df, cost_df, cfg.figure_dir)

    print("=== Holding & Attribution Summary ===")
    for k, v in holding_stats.items():
        print(f"  {k:32s}: {v:+.4f}")
    print("\n=== Yearly Brinson Attribution ===")
    print(yearly_brinson_df.to_string(index=False))
    print("\n=== Top 5 & Bottom 5 Industry Attribution ===")
    print(pd.concat([ind_df.head(5), ind_df.tail(5)]).to_string(index=False))
    print("\n=== Style Exposure (Overall Active Z) ===")
    print(style_df[["Factor", "Overall_Active_Z", "Y2018", "Y2022", "Y2024", "Y2025"]].to_string(index=False))
    print("\n=== Market Regime Applicability ===")
    print(regime_df.to_string(index=False))
    print("\n=== Historical Era Applicability ===")
    print(era_df.to_string(index=False))
    print("\n=== Cost Sensitivity ===")
    print(cost_df.to_string(index=False))


if __name__ == "__main__":
    main()
