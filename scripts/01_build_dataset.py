"""Step 1: Build and audit the cached CSI 500 PIT tensor & market panel."""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import build_and_cache_panel


def main() -> None:
    cfg = ExperimentConfig()
    force = "--force" in sys.argv
    t0 = time.time()
    panel = build_and_cache_panel(cfg, force_rebuild=force)
    dt = time.time() - t0

    # Strict audit checks
    valid_cnts = panel.valid_mask.sum(axis=1)
    lbl_cnts = panel.label_valid.sum(axis=1)
    assert np.isfinite(panel.X_seq).all(), "Non-finite value in X_seq!"
    assert np.isfinite(panel.X_stat).all(), "Non-finite value in X_stat!"
    assert np.isfinite(panel.X_tab).all(), "Non-finite value in X_tab!"
    assert np.isfinite(panel.y_norm).all(), "Non-finite value in y_norm!"
    assert np.isfinite(panel.y_rank).all(), "Non-finite value in y_rank!"
    assert str(panel.signal_dates[-1]) <= "2025-12-31", "2026+ data leaked into panel!"

    # Verify strict temporal ordering: signal_date < exec_date < label_end_date
    for k in range(len(panel.signal_dates)):
        assert panel.signal_cal_idx[k] < panel.exec_cal_idx[k], f"Signal >= exec at k={k}"
        if panel.label_end_cal_idx[k] >= 0:
            assert panel.exec_cal_idx[k] < panel.label_end_cal_idx[k], f"Exec >= label_end at k={k}"

    print("=== TS-CSNet Panel Cache Ready ===")
    print(f"Cache file     : {cfg.cache_path}")
    print(f"Elapsed        : {dt:.2f}s")
    print(f"Calendar range : {panel.cal_dates[0]} .. {panel.cal_dates[-1]} ({len(panel.cal_dates)} days)")
    print(f"Schedule range : {panel.signal_dates[0]} .. {panel.signal_dates[-1]} ({len(panel.signal_dates)} weeks)")
    print(f"Universe size  : {len(panel.stock_codes)} unique historical CSI 500 stocks")
    print(f"X_seq shape    : {panel.X_seq.shape}  (K, M, T=20, C=6)")
    print(f"X_stat shape   : {panel.X_stat.shape}  (K, M, C=8)")
    print(f"X_tab shape    : {panel.X_tab.shape}  (K, M, C=26)")
    print(f"Active stocks/w: min={valid_cnts[valid_cnts > 0].min()}, median={int(np.median(valid_cnts))}, max={valid_cnts.max()}")
    print(f"Labeled/week   : median={int(np.median(lbl_cnts[:-2]))}")
    print("Audit checks   : ALL PASSED (zero NaNs, strict signal < exec < label_end, <= 2025-12-31)")


if __name__ == "__main__":
    main()
