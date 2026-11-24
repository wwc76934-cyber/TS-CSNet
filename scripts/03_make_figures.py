"""Step 3: Generate all 4 defense figures from saved experiment outputs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import load_panel
from tscs_alpha.plots import (
    plot_ablation_overview,
    plot_interpretability_and_cum_ic,
    plot_nav_and_excess,
    plot_yearly_excess,
)


def main() -> None:
    cfg = ExperimentConfig()
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(cfg.output_dir / "ablation_summary.csv")
    nav_df = pd.read_csv(cfg.output_dir / "daily_nav.csv")
    ex_df = pd.read_csv(cfg.output_dir / "daily_excess_nav.csv")
    yearly_df = pd.read_csv(cfg.output_dir / "yearly_breakdown.csv")
    gate_path = cfg.output_dir / "tscsnet_gate_history.csv"
    gate_df = pd.read_csv(gate_path) if gate_path.exists() else pd.DataFrame(columns=["signal_date", "gate_temporal_weight"])

    panel = load_panel(cfg)
    scores_npz = dict(np.load(cfg.output_dir / "oos_scores.npz"))

    plot_ablation_overview(summary_df, cfg.figure_dir / "01_ablation_overview.png")
    plot_nav_and_excess(nav_df, ex_df, cfg.figure_dir / "02_nav_and_excess_curves.png")
    plot_yearly_excess(yearly_df, cfg.figure_dir / "03_yearly_excess_comparison.png")
    plot_interpretability_and_cum_ic(
        panel, cfg, scores_npz, gate_df, cfg.figure_dir / "04_gate_and_ic_interpretability.png"
    )
    print(f"All 4 figures saved to {cfg.figure_dir}")


if __name__ == "__main__":
    main()
