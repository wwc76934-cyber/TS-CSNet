"""Generate a clean, finance-first 4-module strategy workflow diagram (Figure 00)
in PKU minimalist academic style for both the Midterm Report and PPT slides.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200

OUT_PATH = Path(r"D:\QuantProjects\tscs-alpha\figures\00_strategy_architecture_flow.png")
ARTIFACT_PATH = Path(r"C:\Users\18137\.gemini\antigravity\brain\3cec51ec-f1ee-4889-8a19-fbed4398c912\00_strategy_architecture_flow.png")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Top title strip
    ax.text(
        1.5, 33.4,
        "中证500量价与基本面融合增强策略核心流程：从纯 PIT 市场信息到合规组合执行",
        fontsize=13.5, fontweight="bold", color="#1B365D", va="center",
    )
    ax.plot([1.5, 98.5], [31.2, 31.2], color="#8B1A24", linewidth=1.6)

    # 4 Main Module Cards
    cards = [
        {
            "x": 1.5, "w": 21.8, "tag": "01  选股打分", "color": "#1B365D", "bg": "#F4F7FB",
            "title": "多视角相对强弱排序",
            "bullets": [
                "• 6类20日量价走势形态",
                "• 8项公告日纯PIT基本面",
                "• TS-CSNet 横向比较500股",
                "• 无偏排序损失输出打分 s",
            ],
            "foot": "解决：单股孤立建模与MSE错配",
        },
        {
            "x": 26.2, "w": 21.8, "tag": "02  择时与适应", "color": "#2B5B84", "bg": "#F4F8FB",
            "title": "放弃大盘择时·专注风格适应",
            "bullets": [
                "• 合同定位：99%高权益仓位",
                "• 双周调仓 + 因果EMA平滑",
                "• 年度滚动重训（3年窗口）",
                "• 门控层自动平衡量价/基本面",
            ],
            "foot": "解决：牛市踏空与风格迁移滞后",
        },
        {
            "x": 50.9, "w": 21.8, "tag": "03  组合配置", "color": "#8B1A24", "bg": "#FDF5F6",
            "title": "Soft-Tilt 软倾斜指数增强",
            "bullets": [
                "• 锚定中证500流通市值权重",
                "• 头部高分股：最高2.5×超配",
                "• 中间分化股：适度低配缓冲",
                "• 尾部差股(<-1σ)：直接清零",
            ],
            "foot": "解决：集中持股回撤大与排雷不足",
        },
        {
            "x": 75.6, "w": 22.9, "tag": "04  风控与执行", "color": "#1B6B4F", "bg": "#F3FAF7",
            "title": "实盘约束与成本纪律",
            "bullets": [
                "• 涨跌停/停牌/ST硬拦截锁仓",
                "• 主板100股 / 科创板≥200股",
                "• 贪心余钱补手（现金≤0.5%）",
                "• 严格10日隔离期防未来函数",
            ],
            "foot": "解决：回测失真与整手现金拖累",
        },
    ]

    for idx, c in enumerate(cards):
        # Card background
        box = mpatches.FancyBboxPatch(
            (c["x"], 6.2), c["w"], 22.8,
            boxstyle="round,pad=0.3,rounding_size=0.9",
            facecolor=c["bg"], edgecolor="#CBD5E1", linewidth=1.0,
        )
        ax.add_patch(box)

        # Top header bar inside card
        hdr = mpatches.FancyBboxPatch(
            (c["x"], 24.4), c["w"], 4.6,
            boxstyle="round,pad=0.3,rounding_size=0.8",
            facecolor=c["color"], edgecolor="none",
        )
        ax.add_patch(hdr)
        ax.text(c["x"] + 1.1, 26.7, c["tag"], fontsize=10.5, fontweight="bold", color="#FFFFFF", va="center")

        # Subtitle
        ax.text(c["x"] + 1.0, 22.1, c["title"], fontsize=10.0, fontweight="bold", color="#0F172A", va="center")
        ax.plot([c["x"] + 1.0, c["x"] + c["w"] - 1.0], [20.5, 20.5], color="#CBD5E1", linewidth=0.8)

        # Bullets
        for bi, btxt in enumerate(c["bullets"]):
            ax.text(c["x"] + 1.0, 18.6 - bi * 2.2, btxt, fontsize=8.7, color="#334155", va="center")

        # Footer pill
        fbox = mpatches.FancyBboxPatch(
            (c["x"] + 0.7, 6.9), c["w"] - 1.4, 2.7,
            boxstyle="round,pad=0.15,rounding_size=0.5",
            facecolor="#FFFFFF", edgecolor=c["color"], linewidth=0.8,
        )
        ax.add_patch(fbox)
        ax.text(c["x"] + c["w"] / 2.0, 8.25, c["foot"], fontsize=8.0, fontweight="bold", color=c["color"], ha="center", va="center")

        # Arrow to next card
        if idx < len(cards) - 1:
            ax.annotate(
                "",
                xy=(cards[idx + 1]["x"] - 0.55, 17.6),
                xytext=(c["x"] + c["w"] + 0.55, 17.6),
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.35", color="#64748B", lw=1.8),
            )

    # Bottom feedback bar (Iterative Research Loop)
    loop_box = mpatches.FancyBboxPatch(
        (1.5, 1.0), 97.0, 3.3,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=0.9, linestyle="--",
    )
    ax.add_patch(loop_box)
    ax.text(
        50.0, 2.65,
        "闭环定量研究与迭代：9年样本外回测 (2017–2025)  →  持仓 / Brinson / 风格归因  →  行情与成本适用性诊断  →  实盘模拟建仓 (2026-11-24) 与结项复盘",
        fontsize=8.9, fontweight="bold", color="#334155", ha="center", va="center",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    fig.savefig(ARTIFACT_PATH, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("Saved workflow diagram:", OUT_PATH)


if __name__ == "__main__":
    main()
