"""Render single-panel, slide-optimized charts (large fonts, no chart titles,
consistent PKU palette) for the redesigned midterm deck.

Each chart is sized to the exact pixel box it occupies on the 1280x720 slide
(figsize = px / 100), so on-slide text renders at ~11-13pt.
Outputs: figures/slides/*.png and outputs/slide_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tscs_alpha.backtest import get_oos_rebalance_indices
from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import load_panel

OUT_DIR = PROJECT_ROOT / "figures" / "slides"
RES = PROJECT_ROOT / "outputs"

NAVY = "#1B365D"
CRIMSON = "#8B1A24"
GREEN = "#1B6B4F"
STEEL = "#5B7FA6"
LIGHT = "#B4C4D6"
GRAY = "#7B8794"
GOLD = "#C08A3E"
ROSE = "#C49AA3"
INK = "#1E293B"
MUTED = "#475569"
GRID = "#E5EAF0"
SHADE = "#FBEAEA"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED,
    "axes.labelsize": 11,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10.5,
})
DPI = 250

MODEL_LABEL = {
    "M1_Ridge": "M1 线性岭回归",
    "M2_MLP_MSE": "M2 MLP",
    "M3_GRU_MSE": "M3 GRU",
    "M4_GRU_ICRank": "M4 GRU+排序损失",
    "M5_TSCSNet": "M5 本策略（完整）",
}
MODEL_COLOR = {
    "M1_Ridge": LIGHT,
    "M2_MLP_MSE": GOLD,
    "M3_GRU_MSE": ROSE,
    "M4_GRU_ICRank": GREEN,
    "M5_TSCSNet": NAVY,
}


def new_fig(w_px: float, h_px: float):
    return plt.subplots(figsize=(w_px / 100.0, h_px / 100.0))


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / name, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)


def shade_2025(ax, label_y=None) -> None:
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), color=SHADE, zorder=0, lw=0)
    if label_y is not None:
        ax.text(pd.Timestamp("2025-07-01"), label_y, "2025\n高Beta科技牛", ha="center", va="top",
                fontsize=9.5, color=CRIMSON, fontweight="bold")


def year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    stats: dict = {}

    summary = pd.read_csv(RES / "ablation_summary.csv").set_index("Model")
    nav = pd.read_csv(RES / "daily_nav.csv")
    exn = pd.read_csv(RES / "daily_excess_nav.csv")
    yearly = pd.read_csv(RES / "yearly_breakdown.csv")
    dates = pd.to_datetime(nav["date"])

    # ---------------------------------------------------------------- 1. NAV
    fig, ax = new_fig(784, 432)
    shade_2025(ax, label_y=None)
    series = [
        ("B0_CSI500_TR", "中证500全收益基准", GRAY, "--", 1.7),
        ("B1_CSI500_EW", "成分股等权", LIGHT, ":", 1.6),
        ("M1_Ridge", "线性多因子对照 (M1)", STEEL, "-", 1.4),
        ("M5_TSCSNet", "本策略 (M5)", NAVY, "-", 2.6),
    ]
    for col, lab, colr, ls, lw in series:
        ax.plot(dates, nav[col], color=colr, ls=ls, lw=lw, label=lab)
        v = float(nav[col].iloc[-1])
        dy = 7 if col == "B1_CSI500_EW" else (-7 if col == "B0_CSI500_TR" else 0)
        ax.annotate(f"{v:.2f}", xy=(dates.iloc[-1], v), xytext=(6, dy), textcoords="offset points",
                    va="center", fontsize=10.5, color=colr if col != "B1_CSI500_EW" else GRAY, fontweight="bold")
    ax.set_xlim(dates.iloc[0], dates.iloc[-1] + pd.Timedelta(days=150))
    ax.axhline(1.0, color="#94A3B8", lw=0.8)
    ax.set_ylabel("累计净值（起始 = 1.0）")
    year_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    ymax = ax.get_ylim()[1]
    ax.text(pd.Timestamp("2025-07-01"), ymax * 0.985, "2025", ha="center", va="top", fontsize=9.5, color=CRIMSON, fontweight="bold")
    save(fig, "c01_nav.png")
    stats["nav_final"] = {c: float(nav[c].iloc[-1]) for c in ["B0_CSI500_TR", "B1_CSI500_EW", "M1_Ridge", "M5_TSCSNet"]}

    # ------------------------------------------------------------- 2. Excess
    fig, ax = new_fig(784, 432)
    shade_2025(ax)
    ax.axhline(1.0, color="#94A3B8", lw=0.9)
    ax.plot(dates, exn["B1_CSI500_EW"], color=GRAY, ls=":", lw=1.4, label="成分股等权")
    for m in ["M1_Ridge", "M2_MLP_MSE", "M3_GRU_MSE", "M4_GRU_ICRank", "M5_TSCSNet"]:
        lw = 2.7 if m == "M5_TSCSNet" else 1.3
        al = 1.0 if m in ("M5_TSCSNet", "M4_GRU_ICRank") else 0.85
        ax.plot(dates, exn[m], color=MODEL_COLOR[m], lw=lw, alpha=al, label=MODEL_LABEL[m])
    m5 = exn["M5_TSCSNet"].to_numpy()
    ip = int(np.argmax(m5))
    peak_v, peak_d = float(m5[ip]), dates.iloc[ip]
    ax.annotate(f"峰值 {peak_v:.2f}\n({peak_d:%Y-%m})", xy=(peak_d, peak_v), xytext=(-120, 6),
                textcoords="offset points", fontsize=10, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.0))
    ax.annotate(f"{m5[-1]:.2f}", xy=(dates.iloc[-1], m5[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10.5, color=NAVY, fontweight="bold")
    ax.set_xlim(dates.iloc[0], dates.iloc[-1] + pd.Timedelta(days=120))
    ax.set_ylabel("相对超额净值（策略 / 中证500全收益）")
    year_axis(ax)
    ax.legend(loc="upper left", ncol=2, fontsize=10)
    save(fig, "c02_excess.png")
    pre25 = exn.loc[dates < "2025-01-01", "M5_TSCSNet"]
    stats["excess_peak"] = peak_v
    stats["excess_peak_date"] = f"{peak_d:%Y-%m}"
    stats["excess_final"] = float(m5[-1])
    stats["excess_end2024"] = float(pre25.iloc[-1])

    # ------------------------------------------------------------- 3. Yearly
    fig, ax = new_fig(1152, 346)
    y5 = yearly[yearly["Model"] == "M5_TSCSNet"].set_index("Year")
    y1 = yearly[yearly["Model"] == "M1_Ridge"].set_index("Year")
    yrs = list(y5.index)
    x = np.arange(len(yrs))
    w = 0.36
    v5 = y5["Excess_Ret"].to_numpy() * 100
    v1 = y1.loc[yrs, "Excess_Ret"].to_numpy() * 100
    ax.bar(x - w / 2, v1, width=w, color=LIGHT, label="线性多因子对照 (M1)")
    ax.bar(x + w / 2, v5, width=w, color=[NAVY if v >= 0 else CRIMSON for v in v5], label="本策略 (M5)")
    for xi, v in zip(x, v5):
        ax.text(xi + w / 2, v + (0.35 if v >= 0 else -0.35), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10.5, fontweight="bold", color=NAVY if v >= 0 else CRIMSON)
    ax.axhline(0, color="#64748B", lw=0.9)
    bench = y5["CSI500_TR_Ret"].to_numpy() * 100
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n基准 {b:+.1f}%" for y, b in zip(yrs, bench)], fontsize=10)
    ax.set_ylabel("年度超额收益 (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_ylim(min(v5.min(), v1.min()) - 2.2, max(v5.max(), v1.max()) + 2.0)
    ax.legend(handles=[Patch(color=LIGHT, label="线性多因子对照 (M1)"), Patch(color=NAVY, label="本策略 (M5) 跑赢"),
                       Patch(color=CRIMSON, label="本策略 (M5) 跑输")], loc="lower left", ncol=3)
    save(fig, "c03_yearly.png")

    # ------------------------------------------------------- 4/5. Ablation
    models = ["M1_Ridge", "M2_MLP_MSE", "M3_GRU_MSE", "M4_GRU_ICRank", "M5_TSCSNet"]
    xl = ["M1\n线性岭回归", "M2\nMLP", "M3\nGRU", "M4\nGRU+排序", "M5\n完整策略"]
    cols = [LIGHT, LIGHT, LIGHT, STEEL, NAVY]

    fig, ax = new_fig(558, 360)
    vals = summary.loc[models, "RankIC"].to_numpy() * 100
    icir = summary.loc[models, "RankICIR_Ann"].to_numpy()
    ax.bar(np.arange(5), vals, color=cols, width=0.6)
    for i, (v, c) in enumerate(zip(vals, icir)):
        ax.text(i, v + 0.15, f"{v:.2f}%\nICIR {c:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold" if i >= 3 else "normal", color=NAVY if i >= 3 else MUTED)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(xl)
    ax.set_ylim(0, vals.max() * 1.42)
    ax.set_ylabel("样本外双周 RankIC (%)")
    ax.grid(axis="y", color=GRID)
    ax.annotate("", xy=(2.92, vals[3] * 1.27), xytext=(2.05, vals[2] * 1.27),
                arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.8))
    ax.text(2.48, vals[3] * 1.30, "+1.78pp", ha="center", va="bottom", color=CRIMSON, fontsize=10.5, fontweight="bold")
    save(fig, "c04_ablation_ic.png")

    fig, ax = new_fig(558, 360)
    vals = summary.loc[models, "Ann_Excess"].to_numpy() * 100
    irs = summary.loc[models, "IR"].to_numpy()
    ax.bar(np.arange(5), vals, color=cols, width=0.6)
    for i, (v, r) in enumerate(zip(vals, irs)):
        ax.text(i, v + 0.04, f"{v:+.2f}%\nIR {r:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold" if i == 4 else "normal", color=NAVY if i == 4 else MUTED)
    ew = float(summary.loc["B1_CSI500_EW", "Ann_Excess"]) * 100
    ax.axhline(ew, color=GRAY, ls=":", lw=1.3)
    ax.text(4.45, ew + 0.03, f"等权 {ew:+.2f}%", ha="right", va="bottom", fontsize=9.5, color=GRAY)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(xl)
    ax.set_ylim(0, vals.max() * 1.35)
    ax.set_ylabel("年化净超额 vs 中证500全收益 (%)")
    ax.grid(axis="y", color=GRID)
    save(fig, "c05_ablation_excess.png")

    # ------------------------------------------------- 6/7. Cum IC & Decile
    panel = load_panel(cfg)
    rebal = get_oos_rebalance_indices(panel, cfg)
    scores = np.load(RES / "oos_scores.npz")
    rdates = pd.to_datetime([str(panel.signal_dates[k]) for k in rebal])

    fig, ax = new_fig(784, 432)
    shade_2025(ax)
    finals = {}
    for m in models:
        sc = scores[m]
        ics = []
        for k in rebal:
            mk = panel.label_valid[k] & np.isfinite(sc[k])
            if mk.sum() < 50:
                ics.append(0.0)
                continue
            r = spearmanr(sc[k, mk], panel.y_raw[k, mk]).statistic
            ics.append(0.0 if not np.isfinite(r) else float(r))
        cum = np.cumsum(ics)
        finals[m] = float(cum[-1])
        ax.plot(rdates, cum, color=MODEL_COLOR[m], lw=2.7 if m == "M5_TSCSNet" else 1.4, label=MODEL_LABEL[m])
        if m in ("M5_TSCSNet", "M1_Ridge"):
            ax.annotate(f"{cum[-1]:.1f}", xy=(rdates[-1], cum[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10.5, color=MODEL_COLOR[m] if m == "M5_TSCSNet" else STEEL, fontweight="bold")
        if m == "M5_TSCSNet":
            i25 = int(np.searchsorted(rdates.values, np.datetime64("2025-01-01")))
            ax.annotate("2025年累计RankIC\n仍持续上行", xy=(rdates[i25 + 12], cum[i25 + 12]), xytext=(-170, 20),
                        textcoords="offset points", fontsize=10, color=CRIMSON, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.1))
    ax.axhline(0, color="#94A3B8", lw=0.8)
    ax.set_xlim(rdates[0], rdates[-1] + pd.Timedelta(days=120))
    ax.set_ylabel("累计双周 RankIC")
    year_axis(ax)
    ax.legend(loc="upper left", fontsize=10)
    save(fig, "c06_cum_ic.png")
    stats["cum_ic_final"] = finals

    fig, ax = new_fig(662, 376)
    sc = scores["M5_TSCSNet"]
    dec = [[] for _ in range(10)]
    for k in rebal:
        mk = panel.label_valid[k] & np.isfinite(sc[k])
        if mk.sum() < 100:
            continue
        s = sc[k, mk]
        y = panel.y_raw[k, mk]
        ye = y - y.mean()
        for di, ch in enumerate(np.array_split(np.argsort(-s), 10)):
            dec[di].append(float(ye[ch].mean()))
    dv = np.array([np.mean(d) * 26 * 100 for d in dec])
    ax.bar(np.arange(1, 11), dv, color=[GREEN if v >= 0 else CRIMSON for v in dv], width=0.62)
    for i, v in enumerate(dv, start=1):
        ax.text(i, v + (0.35 if v >= 0 else -0.35), f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=10, fontweight="bold", color=GREEN if v >= 0 else CRIMSON)
    ax.axhline(0, color="#64748B", lw=0.9)
    ax.set_xticks(np.arange(1, 11))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    ax.set_xlabel("模型得分十分位（D1 最高 → D10 最低）")
    ax.set_ylabel("年化相对等权超额 (%)")
    ax.set_ylim(dv.min() - 2.8, dv.max() + 2.4)
    ax.grid(axis="y", color=GRID)
    save(fig, "c07_decile.png")
    stats["decile"] = dv.tolist()

    # ----------------------------------------------- 8/9. Brinson & Industry
    by = pd.read_csv(RES / "brinson_yearly_table.csv")
    fig, ax = new_fig(558, 362)
    xx = np.arange(len(by))
    ax.bar(xx - 0.2, by["Allocation"] * 100, width=0.38, color=LIGHT, label="行业配置效应")
    ax.bar(xx + 0.2, by["Selection"] * 100, width=0.38, color=NAVY, label="行业内选股效应")
    ax.plot(xx, by["Total_Gross_Active"] * 100, color=CRIMSON, marker="D", ms=5, lw=0, label="合计（含交互项）")
    ax.axhline(0, color="#64748B", lw=0.9)
    ax.set_xticks(xx)
    ax.set_xticklabels([str(int(y))[2:] + "年" for y in by["Year"]], fontsize=10)
    ax.set_ylabel("年度毛超额贡献 (%)")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="lower left", fontsize=9.5)
    save(fig, "c08_brinson_yearly.png")

    ind = pd.read_csv(RES / "brinson_industry_table.csv")
    sub = pd.concat([ind.head(6), ind.tail(5)]).sort_values("Ann_Total_Contrib")
    fig, ax = new_fig(558, 362)
    v = sub["Ann_Total_Contrib"].to_numpy() * 100
    ax.barh(np.arange(len(sub)), v, color=[NAVY if t >= 0 else CRIMSON for t in v], height=0.62)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["Industry"], fontsize=10.5)
    for i, t in enumerate(v):
        ax.text(t + (0.015 if t >= 0 else -0.015), i, f"{t:+.2f}%", va="center", ha="left" if t >= 0 else "right",
                fontsize=10, color=NAVY if t >= 0 else CRIMSON, fontweight="bold")
    ax.axvline(0, color="#64748B", lw=0.9)
    ax.set_xlim(v.min() - 0.2, v.max() + 0.2)
    ax.set_xlabel("年化超额贡献 (%)")
    ax.grid(axis="x", color=GRID)
    ax.spines["left"].set_visible(False)
    save(fig, "c09_industry.png")

    # ------------------------------------------------------------ 10. Donut
    hs = json.loads((RES / "holding_and_attribution_summary.json").read_text(encoding="utf-8"))
    ow = hs["Ann_Gross_Overweight_Contrib"] * 100
    zr = hs["Ann_Gross_Zeroed_Tail_Contrib"] * 100
    uw_mid = (hs["Ann_Gross_Underweight_Contrib"] - hs["Ann_Gross_Zeroed_Tail_Contrib"]) * 100
    tot = ow + zr + uw_mid
    fig, ax = new_fig(456, 384)
    parts = [ow, uw_mid, zr]
    labs = [
        f"头部超配\n+{ow:.2f}% ({ow / tot:.0%})",
        f"中部低配\n+{uw_mid:.2f}% ({uw_mid / tot:.0%})",
        f"尾部清零\n+{zr:.2f}% ({zr / tot:.0%})",
    ]
    wedges, _ = ax.pie(
        parts, radius=0.78, colors=[NAVY, STEEL, CRIMSON], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=2.2),
    )
    callouts = [
        (1.02, 0.54, "left"),
        (0.95, -0.78, "left"),
        (-1.02, -0.18, "right"),
    ]
    for wd, lb, c, (ox, oy, ha_mode) in zip(wedges, labs, [NAVY, STEEL, CRIMSON], callouts):
        ang = np.deg2rad((wd.theta2 + wd.theta1) / 2.0)
        xr, yr = np.cos(ang), np.sin(ang)
        ax.annotate(
            lb, xy=(0.78 * xr, 0.78 * yr), xytext=(ox, oy), ha=ha_mode, va="center",
            fontsize=10.2, fontweight="bold", color=c,
            arrowprops=dict(arrowstyle="-", color=c, lw=1.1, connectionstyle="arc3,rad=0.08"),
        )
    ax.text(0, 0.08, "年化毛超额", ha="center", va="center", fontsize=9.8, color=MUTED)
    ax.text(0, -0.12, f"+{tot:.2f}%", ha="center", va="center", fontsize=15.0, color=NAVY, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.35, 1.35)
    save(fig, "c10_donut.png")
    stats["contrib"] = {"overweight": ow, "mid_underweight": uw_mid, "zeroed": zr, "total": tot}

    # ------------------------------------------------------------ 11. Style
    st = pd.read_csv(RES / "style_exposure_table.csv")
    lab_map = {"ep_ttm": "盈利收益率 EP", "bp_mrq": "账面市值比 BP", "roe_ttm": "隐含 ROE", "roe_chg_60d": "60日 ROE 改善",
               "log_float_mv": "流通市值", "mom_60d": "60日动量", "vol_60d": "60日波动率", "turn_cv_20d": "换手率变异系数",
               "mean_ret20": "20日短期收益", "mean_turn20": "20日平均换手"}
    st = st.sort_values("Overall_Active_Z")
    fig, ax = new_fig(664, 516)
    yy = np.arange(len(st))
    ax.barh(yy + 0.2, st["Overall_Active_Z"], height=0.38, color=STEEL, label="9年平均暴露")
    ax.barh(yy - 0.2, st["Y2025"], height=0.38, color=CRIMSON, label="2025年暴露")
    for i, (a, b) in enumerate(zip(st["Overall_Active_Z"], st["Y2025"])):
        ax.text(b + (0.012 if b >= 0 else -0.012), i - 0.2, f"{b:+.2f}", va="center", ha="left" if b >= 0 else "right",
                fontsize=9.5, color=CRIMSON, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([lab_map.get(f, f) for f in st["Factor"]], fontsize=10.5)
    ax.axvline(0, color="#64748B", lw=0.9)
    ax.set_xlim(st[["Overall_Active_Z", "Y2025"]].min().min() - 0.16, st[["Overall_Active_Z", "Y2025"]].max().max() + 0.14)
    ax.set_xlabel("相对基准的主动风格暴露（截面 Z-Score）")
    ax.grid(axis="x", color=GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right")
    save(fig, "c11_style.png")

    # ----------------------------------------------------------- 12. Regime
    rg = pd.read_csv(RES / "regime_applicability.csv")
    names = ["单边牛市", "震荡市", "下跌熊市", "高波动", "低波动"]
    fig, ax = new_fig(784, 432)
    v = rg["Ann_Excess_Arith"].to_numpy() * 100
    ax.bar(np.arange(len(v)), v, color=[GREEN if t >= 0 else CRIMSON for t in v], width=0.56)
    for i, (t, ir) in enumerate(zip(v, rg["IR"])):
        ax.text(i, t + (0.3 if t >= 0 else -0.3), f"{t:+.2f}%\nIR {ir:+.2f}", ha="center", va="bottom" if t >= 0 else "top",
                fontsize=10.5, fontweight="bold", color=GREEN if t >= 0 else CRIMSON)
    ax.axhline(0, color="#64748B", lw=0.9)
    ax.axvline(2.5, color="#CBD5E1", lw=1.0, ls="--")
    ax.text(1.0, v.max() + 1.6, "按 60 日指数涨跌划分", ha="center", fontsize=10, color=MUTED)
    ax.text(3.5, v.max() + 1.6, "按 20 日波动率划分", ha="center", fontsize=10, color=MUTED)
    ax.set_xticks(np.arange(len(v)))
    ax.set_xticklabels([f"{n}\n(占 {s:.0%} 交易日)" for n, s in zip(names, rg["Day_Share"])], fontsize=10.5)
    ax.set_ylabel("年化超额收益 (%)")
    ax.set_ylim(v.min() - 2.6, v.max() + 2.8)
    ax.grid(axis="y", color=GRID)
    save(fig, "c12_regime.png")

    # -------------------------------------------------------- 13/14. Era & Cost
    era = pd.read_csv(RES / "era_applicability.csv")
    fig, ax = new_fig(558, 360)
    ev = era["Ann_Excess"].to_numpy() * 100
    ax.bar(np.arange(4), ev, color=[NAVY if t >= 0 else CRIMSON for t in ev], width=0.52)
    for i, t in enumerate(ev):
        ax.text(i, t + (0.3 if t >= 0 else -0.3), f"{t:+.2f}%", ha="center", va="bottom" if t >= 0 else "top",
                fontsize=10.5, fontweight="bold", color=NAVY if t >= 0 else CRIMSON)
    ax.axhline(0, color="#64748B", lw=0.9)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["2017–18\n去杠杆熊市", "2019–21\n结构性牛市", "2022–24\n熊市/微盘震荡", "2025\n高Beta科技牛"], fontsize=10)
    ax.set_ylabel("年化净超额 (%)")
    ax.set_ylim(ev.min() - 2.5, ev.max() + 3.5)
    ax.grid(axis="y", color=GRID)
    ax2 = ax.twinx()
    ic = era["Mean_RankIC"].to_numpy() * 100
    ax2.plot(np.arange(4), ic, color=GOLD, marker="o", lw=2.2, ms=7, zorder=5)
    ic_offsets = [(0, 13), (0, -17), (0, 13), (0, 15)]
    for i, (t, (dx, dy)) in enumerate(zip(ic, ic_offsets)):
        ax2.annotate(
            f"IC {t:.1f}%", xy=(i, t), xytext=(dx, dy), textcoords="offset points",
            ha="center", va="center", fontsize=9.2, color=GOLD, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=GOLD, lw=0.9, alpha=0.96),
            zorder=6,
        )
    ax2.set_ylim(0, 14)
    ax2.set_ylabel("期均 RankIC (%)", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(GOLD)
    ax2.grid(False)
    save(fig, "c13_era.png")

    cost = pd.read_csv(RES / "cost_applicability.csv")
    fig, ax = new_fig(558, 360)
    cv = cost["Ann_Excess"].to_numpy() * 100
    shades = ["#9FB3C8", "#7895B2", NAVY, "#4A6A8F", "#6B8099"]
    ax.bar(np.arange(5), cv, color=shades, width=0.56)
    for i, t in enumerate(cv):
        ax.text(i, t + 0.05, f"{t:+.2f}%", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["0bp\n理论上限", "10bp\n低费机构", "A股真实\n分段成本", "20bp\n较高冲击", "30bp\n极端摩擦"], fontsize=10)
    ax.set_ylabel("年化净超额 (%)")
    ax.set_ylim(0, cv.max() * 1.25)
    ax.grid(axis="y", color=GRID)
    save(fig, "c14_cost.png")

    # ----------------------------------------------------- 15/16. Live paper
    li = pd.read_csv(RES / "paper_portfolio_industry.csv").head(12).iloc[::-1]
    fig, ax = new_fig(784, 432)
    yy = np.arange(len(li))
    ax.barh(yy + 0.2, li["Actual_Weight"] * 100, height=0.38, color=NAVY, label="模拟组合权重")
    ax.barh(yy - 0.2, li["Bench_Weight"] * 100, height=0.38, color=LIGHT, label="基准底仓权重")
    xmax = float(li[["Actual_Weight", "Bench_Weight"]].max().max() * 100)
    for i, (a, aw) in enumerate(zip(li["Actual_Weight"] * 100, li["Active_Weight"] * 100)):
        ax.text(xmax + 1.0, i, f"{aw:+.2f}%", va="center", ha="left", fontsize=10,
                color=GREEN if aw >= 0 else CRIMSON, fontweight="bold")
    ax.text(xmax + 1.0, len(li) - 0.3, "主动偏离", ha="left", va="bottom", fontsize=10, color=MUTED, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels(li["Industry"], fontsize=10.5)
    ax.set_xlim(0, xmax + 4.2)
    ax.set_xlabel("权重 (%)")
    ax.grid(axis="x", color=GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, 1.10), ncol=2)
    save(fig, "c15_live_industry.png")

    # Gate dynamic curve for Slide 08
    gate_df = pd.read_csv(RES / "tscsnet_gate_history.csv")
    fig, ax = new_fig(520, 210)
    gd = pd.to_datetime(gate_df["signal_date"])
    gv = gate_df["gate_temporal_weight"].to_numpy()
    gs = pd.Series(gv).rolling(8, min_periods=1).mean().to_numpy()
    ax.plot(gd, gv, color=LIGHT, lw=0.9, alpha=0.6)
    ax.plot(gd, gs, color=NAVY, lw=2.2, label="8周平滑门控均值 g")
    ax.axhline(0.5, color=CRIMSON, ls="--", lw=1.0, label="量价/基本面均衡线 (0.50)")
    ax.set_ylim(0.41, 0.54)
    ax.set_ylabel("量价权重 g", fontsize=9.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="upper right", fontsize=9.0)
    save(fig, "c17_gate.png")

    pf = pd.read_csv(RES / "paper_portfolio_live_full.csv").sort_values("Actual_Weight", ascending=False).head(15).iloc[::-1]
    fig, ax = new_fig(536, 496)
    yy = np.arange(len(pf))
    ax.barh(yy, pf["Actual_Weight"] * 100, color=[NAVY if r <= 50 else STEEL for r in pf["Rank"]], height=0.62)
    for i, (w_, r, b) in enumerate(zip(pf["Actual_Weight"] * 100, pf["Rank"], pf["Bench_Weight"] * 100)):
        ax.text(w_ + 0.03, i, f"{w_:.2f}%  · 排名 #{int(r)}", va="center", fontsize=10, color=INK)
        ax.plot([b, b], [i - 0.36, i + 0.36], color=CRIMSON, lw=1.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{n}  {c[3:]}  ({ind})" for n, c, ind in zip(pf["Name"], pf["Code"], pf["Industry"])], fontsize=10)
    ax.set_xlim(0, float(pf["Actual_Weight"].max() * 100) + 0.75)
    ax.set_xlabel("计划建仓权重 (%)")
    ax.grid(axis="x", color=GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=NAVY, label="得分排名前 50"), Patch(color=STEEL, label="基准底仓型持仓"),
                       plt.Line2D([0], [0], color=CRIMSON, lw=1.6, label="基准底仓权重")], loc="lower right")
    save(fig, "c16_live_top15.png")

    (RES / "slide_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Saved slide charts to", OUT_DIR)


if __name__ == "__main__":
    main()
