"""Publication-ready matplotlib charts for the TS-CSNet course defense."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .backtest import get_oos_rebalance_indices
from .config import ExperimentConfig
from .dataset import PanelData

# Configure clean Chinese + English font stack on Windows
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

COLORS: Dict[str, str] = {
    "B0_CSI500_TR": "#6c757d",
    "B1_CSI500_EW": "#adb5bd",
    "M1_Ridge": "#4e79a7",
    "M2_MLP_MSE": "#f28e2b",
    "M3_GRU_MSE": "#e15759",
    "M4_GRU_ICRank": "#59a14f",
    "M5_TSCSNet": "#1f3b73",
}

LABELS: Dict[str, str] = {
    "B0_CSI500_TR": "中证500全收益 (B0)",
    "B1_CSI500_EW": "中证500等权 (B1)",
    "M1_Ridge": "M1 线性岭回归",
    "M2_MLP_MSE": "M2 MLP (MSE)",
    "M3_GRU_MSE": "M3 GRU (MSE)",
    "M4_GRU_ICRank": "M4 GRU (IC+排序)",
    "M5_TSCSNet": "M5 TS-CSNet (主力)",
}


def plot_ablation_overview(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 1: Progressive ablation bar comparison (RankIC/ICIR & Excess/IR)."""
    sub = summary_df[summary_df["Model"].isin(["M1_Ridge", "M2_MLP_MSE", "M3_GRU_MSE", "M4_GRU_ICRank", "M5_TSCSNet"])].copy()
    short_names = ["M1\nRidge", "M2\nMLP\n(MSE)", "M3\nGRU\n(MSE)", "M4\nGRU\n(IC+Rank)", "M5\nTS-CSNet\n(主力)"]
    bar_colors = [COLORS[m] for m in sub["Model"]]
    x = np.arange(len(sub))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Panel A: RankIC (%) with RankICIR annotations
    ax = axes[0]
    vals_ic = sub["RankIC"].to_numpy() * 100.0
    icirs = sub["RankICIR_Ann"].to_numpy()
    bars = ax.bar(x, vals_ic, color=bar_colors, width=0.58, edgecolor="white", linewidth=1.0)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylabel("双周样本外 RankIC (%)", fontsize=10.5)
    ax.set_title("递进消融实验（一）：选股信号预测力 (2017–2025)", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    y_span = max(0.5, float(np.nanmax(np.abs(vals_ic))))
    ax.set_ylim(min(0.0, float(np.nanmin(vals_ic)) - 0.18 * y_span), float(np.nanmax(vals_ic)) + 0.28 * y_span)
    for bar, v, icir in zip(bars, vals_ic, icirs):
        va = "bottom" if v >= 0 else "top"
        offset = 0.04 * y_span if v >= 0 else -0.04 * y_span
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + offset,
            f"{v:+.2f}%\n(ICIR {icir:+.2f})",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold" if bar == bars[-1] else "normal",
        )

    # Panel B: Annualized Net Excess Return (%) with IR annotations
    ax = axes[1]
    vals_ex = sub["Ann_Excess"].to_numpy() * 100.0
    irs = sub["IR"].to_numpy()
    bars2 = ax.bar(x, vals_ex, color=bar_colors, width=0.58, edgecolor="white", linewidth=1.0)
    ax.axhline(0, color="#444", linewidth=0.8)
    # Add horizontal reference line for B1_CSI500_EW
    ew_row = summary_df[summary_df["Model"] == "B1_CSI500_EW"]
    if len(ew_row):
        ew_ex = float(ew_row["Ann_Excess"].iloc[0]) * 100.0
        ax.axhline(ew_ex, color="#777", linestyle=":", linewidth=1.3, label=f"等权基准 B1 ({ew_ex:+.2f}%)")
        ax.legend(loc="upper left", frameon=True, fontsize=8.8)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylabel("年化净超额收益 vs 中证500全收益 (%)", fontsize=10.5)
    ax.set_title("递进消融实验（二）：扣费后实盘超额收益与 IR", fontsize=11.5, fontweight="bold", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    y_span2 = max(2.0, float(np.nanmax(np.abs(vals_ex))))
    ax.set_ylim(min(-1.0, float(np.nanmin(vals_ex)) - 0.22 * y_span2), float(np.nanmax(vals_ex)) + 0.30 * y_span2)
    for bar, v, ir in zip(bars2, vals_ex, irs):
        va = "bottom" if v >= 0 else "top"
        offset = 0.04 * y_span2 if v >= 0 else -0.04 * y_span2
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + offset,
            f"{v:+.2f}%\n(IR {ir:+.2f})",
            ha="center",
            va=va,
            fontsize=8.5,
            fontweight="bold" if bar == bars2[-1] else "normal",
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_nav_and_excess(nav_df: pd.DataFrame, ex_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 2: Daily cumulative NAV & cumulative relative excess NAV (2017-2025)."""
    dates = pd.to_datetime(nav_df["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.6), sharex=True)

    plot_models = ["B0_CSI500_TR", "B1_CSI500_EW", "M1_Ridge", "M2_MLP_MSE", "M3_GRU_MSE", "M4_GRU_ICRank", "M5_TSCSNet"]

    # Top panel: Absolute NAV
    ax = axes[0]
    for m in plot_models:
        if m not in nav_df.columns:
            continue
        lw = 2.4 if m == "M5_TSCSNet" else (1.7 if m in ("M4_GRU_ICRank", "B0_CSI500_TR") else 1.3)
        ls = "--" if m.startswith("B") else "-"
        alpha = 1.0 if m in ("M5_TSCSNet", "M4_GRU_ICRank", "B0_CSI500_TR") else 0.8
        ax.plot(dates, nav_df[m], label=LABELS[m], color=COLORS[m], linewidth=lw, linestyle=ls, alpha=alpha)
    ax.set_ylabel("累计净值 (起始 = 1.0)", fontsize=10.5)
    ax.set_title("中证 500 样本外累计净值走势对比 (2017–2025，扣真实双边成本)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=3, fontsize=8.8, frameon=True)

    # Bottom panel: Relative Excess NAV vs H00905
    ax = axes[1]
    ax.axhline(1.0, color="#666", linestyle="--", linewidth=1.0)
    for m in plot_models[1:]:
        if m not in ex_df.columns:
            continue
        lw = 2.4 if m == "M5_TSCSNet" else (1.7 if m == "M4_GRU_ICRank" else 1.3)
        ls = "--" if m.startswith("B") else "-"
        alpha = 1.0 if m in ("M5_TSCSNet", "M4_GRU_ICRank") else 0.82
        ax.plot(dates, ex_df[m], label=LABELS[m], color=COLORS[m], linewidth=lw, linestyle=ls, alpha=alpha)
    ax.set_ylabel("相对超额净值 (策略 / 中证500全收益)", fontsize=10.5)
    ax.set_title("相对中证 500 全收益指数累计超额曲线", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_excess(yearly_df: pd.DataFrame, out_path: Path) -> None:
    """Figure 3: Calendar-year net excess return bar chart."""
    chosen = ["M1_Ridge", "M2_MLP_MSE", "M4_GRU_ICRank", "M5_TSCSNet"]
    years = sorted(yearly_df["Year"].unique().tolist())
    x = np.arange(len(years))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12.2, 4.8))
    for i, m in enumerate(chosen):
        sub = yearly_df[yearly_df["Model"] == m].set_index("Year")
        vals = [float(sub.loc[y, "Excess_Ret"]) * 100.0 for y in years]
        offset = (i - (len(chosen) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=LABELS[m], color=COLORS[m], edgecolor="white", linewidth=0.6)

    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], fontsize=10)
    ax.set_ylabel("分年度净超额收益 (%)", fontsize=10.5)
    ax.set_title("2017–2025 分年度相对中证 500 全收益超额表现对比", fontsize=12, fontweight="bold", pad=24)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.11), ncol=4, frameon=False, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_interpretability_and_cum_ic(
    panel: PanelData,
    cfg: ExperimentConfig,
    scores_npz: Dict[str, np.ndarray],
    gate_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figure 4: Cumulative OOS RankIC + TS-CSNet 10-Decile Monotonicity & Gate Dynamics."""
    rebal_idx = get_oos_rebalance_indices(panel, cfg)
    rebal_dates = pd.to_datetime([str(panel.signal_dates[k]) for k in rebal_idx])

    fig = plt.figure(figsize=(12.4, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    # Top row (full width): Cumulative biweekly OOS RankIC
    ax_top = fig.add_subplot(gs[0, :])
    for m in cfg.model_names:
        if m not in scores_npz:
            continue
        sc = scores_npz[m]
        ics = []
        for k in rebal_idx:
            mask = panel.label_valid[k] & np.isfinite(sc[k])
            if int(mask.sum()) < 50:
                ics.append(0.0)
                continue
            r = spearmanr(sc[k, mask], panel.y_raw[k, mask]).statistic
            ics.append(0.0 if not np.isfinite(r) else float(r))
        cum_ic = np.cumsum(ics)
        lw = 2.4 if m == "M5_TSCSNet" else (1.8 if m == "M4_GRU_ICRank" else 1.3)
        ax_top.plot(rebal_dates, cum_ic, label=f"{LABELS[m]} (累计 {cum_ic[-1]:+.2f})", color=COLORS[m], linewidth=lw)

    ax_top.axhline(0, color="#666", linestyle="--", linewidth=0.9)
    ax_top.set_ylabel("累计双周 RankIC", fontsize=10.5)
    ax_top.set_title("样本外双周 RankIC 累计曲线（M4/M5 引入无偏排序损失后斜率显著陡峭且全周期平滑）", fontsize=11.5, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.35)
    ax_top.legend(loc="upper left", ncol=2, fontsize=8.8, frameon=True)
    ax_top.xaxis.set_major_locator(mdates.YearLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Bottom-left: M5_TSCSNet 10-decile monotonicity bar chart
    ax_dec = fig.add_subplot(gs[1, 0])
    if "M5_TSCSNet" in scores_npz:
        sc = scores_npz["M5_TSCSNet"]
        dec = [[] for _ in range(10)]
        for k in rebal_idx:
            mask = panel.label_valid[k] & np.isfinite(sc[k])
            if int(mask.sum()) < 100:
                continue
            s = sc[k, mask]
            y = panel.y_raw[k, mask]
            y_ex = y - y.mean()
            order = np.argsort(-s)
            for d_i, chunk in enumerate(np.array_split(order, 10)):
                dec[d_i].append(float(y_ex[chunk].mean()))
        dec_ann = np.array([np.mean(d) * 26.0 * 100.0 for d in dec])
        bar_c = ["#1f3b73" if v >= 0 else "#d9534f" for v in dec_ann]
        ax_dec.bar(np.arange(1, 11), dec_ann, color=bar_c, width=0.62, edgecolor="white")
        ax_dec.axhline(0, color="#444", linewidth=0.9)
        ax_dec.set_xticks(np.arange(1, 11))
        ax_dec.set_xticklabels([f"D{i}" for i in range(1, 11)], fontsize=9.0)
        ax_dec.set_ylabel("年化分层超额收益 (%)", fontsize=10.0)
        ax_dec.set_title("M5 TS-CSNet 十分位单调性 (D1最优 → D10最差)", fontsize=10.8, fontweight="bold")
        ax_dec.grid(axis="y", linestyle="--", alpha=0.35)
        for i, v in enumerate(dec_ann, start=1):
            ax_dec.text(i, v + (0.45 if v >= 0 else -0.65), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=7.8)
        ax_dec.set_ylim(float(dec_ann.min()) - 2.6, float(dec_ann.max()) + 2.2)

    # Bottom-right: TS-CSNet learned temporal vs fundamental gate weight g_t
    ax_g = fig.add_subplot(gs[1, 1])
    if len(gate_df) > 0:
        g_dates = pd.to_datetime(gate_df["signal_date"])
        g_vals = gate_df["gate_temporal_weight"].to_numpy()
        g_smooth = pd.Series(g_vals).rolling(8, min_periods=1).mean().to_numpy()
        ax_g.plot(g_dates, g_vals, color="#9ab3d5", alpha=0.45, linewidth=0.9, label="单周门控均值 g_t")
        ax_g.plot(g_dates, g_smooth, color="#1f3b73", linewidth=2.1, label="8周平滑门控均值")
        ax_g.axhline(0.5, color="#888", linestyle="--", linewidth=1.0, label="均衡基准 (0.5)")
        ax_g.set_ylim(max(0.0, float(g_vals.min()) - 0.05), min(1.0, float(g_vals.max()) + 0.06))
        ax_g.set_ylabel("时序量价门控权重 g", fontsize=10.0)
        ax_g.set_title("门控融合动态权重 (g>0.5偏量价, g<0.5偏基本面)", fontsize=10.8, fontweight="bold")
        ax_g.grid(True, linestyle="--", alpha=0.35)
        ax_g.legend(loc="upper right", fontsize=8.0, frameon=True)
        ax_g.xaxis.set_major_locator(mdates.YearLocator(2))
        ax_g.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
