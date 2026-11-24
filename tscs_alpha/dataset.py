"""Read-only CSI 500 PIT data loader and cross-sectional tensor panel builder.

All features at signal_date t are computed strictly from data available on or
before date t close. All labels are exact open-to-open total returns from
execution_date[k] to execution_date[k + horizon_steps], with explicit
label_end_dates recorded to enforce a strict purge gap during rolling training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ExperimentConfig


@dataclass
class PanelData:
    """In-memory container for the cached CSI 500 learning & backtest panel."""

    # Cross-sectional weekly schedule arrays (length K)
    signal_dates: np.ndarray        # (K,) str 'YYYY-MM-DD'
    exec_dates: np.ndarray          # (K,) str 'YYYY-MM-DD'
    label_end_dates: np.ndarray     # (K,) str 'YYYY-MM-DD' ('' if unavailable)
    signal_cal_idx: np.ndarray      # (K,) int32 index into cal_dates
    exec_cal_idx: np.ndarray        # (K,) int32 index into cal_dates
    label_end_cal_idx: np.ndarray   # (K,) int32 index into cal_dates (-1 if unavailable)

    # Cross-sectional tensors (K, M, ...)
    member_idx: np.ndarray          # (K, M) int32 stock index in 0..S-1 (-1 pad)
    valid_mask: np.ndarray          # (K, M) bool active constituent with data
    label_valid: np.ndarray         # (K, M) bool valid_mask & valid future open-to-open return
    X_seq: np.ndarray               # (K, M, T, 6) float32
    X_stat: np.ndarray              # (K, M, 8) float32
    X_tab: np.ndarray               # (K, M, 26) float32 (fair tabular baseline features)
    y_raw: np.ndarray               # (K, M) float32 raw future horizon return
    y_norm: np.ndarray              # (K, M) float32 cross-sectionally standardized return target
    y_rank: np.ndarray              # (K, M) float32 cross-sectionally rank-normalized target

    # Daily market arrays for exact open-to-open / daily mark-to-market backtest
    cal_dates: np.ndarray           # (D,) str 'YYYY-MM-DD'
    stock_codes: np.ndarray         # (S,) str
    hfq_open: np.ndarray            # (D, S) float32
    hfq_close: np.ndarray           # (D, S) float32 (forward-filled for suspended MTM)
    can_buy: np.ndarray             # (D, S) bool
    can_sell: np.ndarray            # (D, S) bool
    bench_h00905: np.ndarray        # (D,) float64 CSI 500 Total Return index
    bench_000905: np.ndarray        # (D,) float64 CSI 500 Price index


def _cs_winsor_zscore(arr: np.ndarray, valid: np.ndarray, q_lo: float = 0.01, q_hi: float = 0.99) -> np.ndarray:
    """Cross-sectional quantile winsorization + z-score standardization.

    Invalid or NaN entries are mapped to 0.0 (cross-sectional neutral value).
    """
    out = np.zeros_like(arr, dtype=np.float32)
    mask = valid & np.isfinite(arr)
    n = int(mask.sum())
    if n < 5:
        return out
    vals = arr[mask].astype(np.float64)
    lo, hi = np.quantile(vals, [q_lo, q_hi])
    vals = np.clip(vals, lo, hi)
    mean = vals.mean()
    std = vals.std()
    if std < 1e-8:
        return out
    out[mask] = ((vals - mean) / std).astype(np.float32)
    return out


def _cs_rank_zscore(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional uniform rank standardized to mean 0, std 1 (symmetric, outlier-free)."""
    out = np.zeros_like(arr, dtype=np.float32)
    mask = valid & np.isfinite(arr)
    n = int(mask.sum())
    if n < 5:
        return out
    vals = arr[mask].astype(np.float64)
    ranks = pd.Series(vals).rank(method="average").to_numpy(dtype=np.float64)
    ranks = (ranks - ranks.mean()) / (ranks.std() + 1e-8)
    out[mask] = ranks.astype(np.float32)
    return out


def _limit_ratio(code: str, date_ints: np.ndarray, is_st: np.ndarray) -> np.ndarray:
    """Compute exact A-share daily price limit ratio per stock and trading date."""
    limits = np.full(len(date_ints), 0.10, dtype=np.float64)
    if code.startswith("sh.688"):
        limits[:] = 0.20
    elif code.startswith("sz.300") or code.startswith("sz.301"):
        limits[date_ints >= 20200824] = 0.20
    limits[is_st == 1] = 0.05
    return limits


def build_and_cache_panel(cfg: ExperimentConfig, force_rebuild: bool = False) -> PanelData:
    """Build the CSI 500 tensor & daily market cache from local raw parquet files."""
    if cfg.cache_path.exists() and not force_rebuild:
        return load_panel(cfg)

    raw_root = cfg.source_data_dir
    if not raw_root.exists():
        raise FileNotFoundError(f"Source data directory not found: {raw_root}")

    # 1. Load trading calendar, schedule, and benchmarks
    cal_df = pd.read_parquet(raw_root / "calendar.parquet")
    cal_ts = pd.to_datetime(cal_df["date"]).sort_values().reset_index(drop=True)
    cal_dates = cal_ts.dt.strftime("%Y-%m-%d").to_numpy(dtype="<U10")
    cal_date_ints = cal_ts.dt.strftime("%Y%m%d").astype(np.int64).to_numpy()
    date_to_idx: Dict[str, int] = {d: i for i, d in enumerate(cal_dates)}
    n_days = len(cal_dates)

    sched_df = pd.read_parquet(raw_root / "schedule.parquet").sort_values("signal_date").reset_index(drop=True)
    sched_sig = pd.to_datetime(sched_df["signal_date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype="<U10")
    sched_exe = pd.to_datetime(sched_df["execution_date"]).dt.strftime("%Y-%m-%d").to_numpy(dtype="<U10")

    # Keep only schedule dates present in calendar (and strictly <= 2025-12-31)
    keep_sched = [
        i for i in range(len(sched_sig))
        if sched_sig[i] in date_to_idx and sched_exe[i] in date_to_idx and sched_sig[i] <= "2025-12-31"
    ]
    signal_dates = sched_sig[keep_sched]
    exec_dates = sched_exe[keep_sched]
    signal_cal_idx = np.array([date_to_idx[d] for d in signal_dates], dtype=np.int32)
    exec_cal_idx = np.array([date_to_idx[d] for d in exec_dates], dtype=np.int32)
    n_sched = len(signal_dates)

    h00905_df = pd.read_parquet(raw_root / "H00905.parquet")
    h00905_df["d"] = pd.to_datetime(h00905_df["date"]).dt.strftime("%Y-%m-%d")
    bench_h00905 = h00905_df.set_index("d")["close"].reindex(cal_dates).ffill().to_numpy(dtype=np.float64)

    i000905_df = pd.read_parquet(raw_root / "000905.parquet")
    i000905_df["d"] = pd.to_datetime(i000905_df["date"]).dt.strftime("%Y-%m-%d")
    bench_000905 = i000905_df.set_index("d")["close"].reindex(cal_dates).ffill().to_numpy(dtype=np.float64)

    # 2. Load PIT weekly constituent snapshots
    members_dir = raw_root / "members"
    member_lists: List[List[str]] = []
    all_codes_set = set()
    for d in signal_dates:
        m_path = members_dir / f"{d}.parquet"
        if not m_path.exists():
            raise FileNotFoundError(f"Missing PIT member file: {m_path}")
        codes = pd.read_parquet(m_path)["code"].astype(str).unique().tolist()
        member_lists.append(codes)
        all_codes_set.update(codes)

    raw_dir = raw_root / "bars_raw"
    hfq_dir = raw_root / "bars_hfq"
    val_dir = raw_root / "valuations"

    available_codes = sorted(
        c for c in all_codes_set
        if (raw_dir / f"{c}.parquet").exists()
        and (hfq_dir / f"{c}.parquet").exists()
        and (val_dir / f"{c}.parquet").exists()
    )
    code_to_idx: Dict[str, int] = {c: i for i, c in enumerate(available_codes)}
    n_stocks = len(available_codes)

    # 3. Preallocate daily feature & market matrices (D, S)
    hfq_open = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    hfq_close_raw = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    hfq_close_mtm = np.full((n_days, n_stocks), np.nan, dtype=np.float32)
    can_buy = np.zeros((n_days, n_stocks), dtype=bool)
    can_sell = np.zeros((n_days, n_stocks), dtype=bool)

    # 6 daily sequence channels (D, S, 6)
    daily_seq = np.full((n_days, n_stocks, len(cfg.seq_features)), np.nan, dtype=np.float32)
    # 6 daily static PIT channels (D, S, 6)
    daily_stat = np.full((n_days, n_stocks, len(cfg.static_features)), np.nan, dtype=np.float32)

    cal_index = pd.Index(cal_dates, name="d")

    for s_idx, code in enumerate(available_codes):
        rdf = pd.read_parquet(raw_dir / f"{code}.parquet")
        hdf = pd.read_parquet(hfq_dir / f"{code}.parquet")
        vdf = pd.read_parquet(val_dir / f"{code}.parquet")

        rdf["d"] = pd.to_datetime(rdf["date"]).dt.strftime("%Y-%m-%d")
        hdf["d"] = pd.to_datetime(hdf["date"]).dt.strftime("%Y-%m-%d")
        vdf["d"] = pd.to_datetime(vdf["date"]).dt.strftime("%Y-%m-%d")

        rdf = rdf.drop_duplicates("d").set_index("d").reindex(cal_index)
        hdf = hdf.drop_duplicates("d").set_index("d").reindex(cal_index)
        vdf = vdf.drop_duplicates("d").set_index("d").reindex(cal_index)

        ro = pd.to_numeric(rdf["open"], errors="coerce").to_numpy(dtype=np.float64)
        rc = pd.to_numeric(rdf["close"], errors="coerce").to_numpy(dtype=np.float64)
        rh = pd.to_numeric(rdf["high"], errors="coerce").to_numpy(dtype=np.float64)
        rl = pd.to_numeric(rdf["low"], errors="coerce").to_numpy(dtype=np.float64)
        rpre = pd.to_numeric(rdf["preclose"], errors="coerce").to_numpy(dtype=np.float64)
        rvol = pd.to_numeric(rdf["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        ramt = pd.to_numeric(rdf["amount"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        rturn = pd.to_numeric(rdf["turn"], errors="coerce").to_numpy(dtype=np.float64)
        tstat = pd.to_numeric(rdf["tradestatus"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        is_st = pd.to_numeric(rdf["is_st"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)

        ho = pd.to_numeric(hdf["open"], errors="coerce").to_numpy(dtype=np.float64)
        hc = pd.to_numeric(hdf["close"], errors="coerce").to_numpy(dtype=np.float64)

        pe = pd.to_numeric(vdf["pe_ttm"], errors="coerce").to_numpy(dtype=np.float64)
        pb = pd.to_numeric(vdf["pb_mrq"], errors="coerce").to_numpy(dtype=np.float64)
        fmv = pd.to_numeric(vdf["float_mv"], errors="coerce").to_numpy(dtype=np.float64)

        # Backtest market prices & tradability flags
        hfq_open[:, s_idx] = ho.astype(np.float32)
        hfq_close_raw[:, s_idx] = hc.astype(np.float32)
        hc_ffill = pd.Series(hc).ffill().to_numpy(dtype=np.float32)
        hfq_close_mtm[:, s_idx] = hc_ffill

        lim = _limit_ratio(code, cal_date_ints, is_st)
        valid_pre = np.isfinite(rpre) & (rpre > 0) & np.isfinite(ro) & (ro > 0)
        up_px = np.round(rpre * (1.0 + lim), 2)
        dn_px = np.round(rpre * (1.0 - lim), 2)
        is_up_open = valid_pre & (ro >= up_px - 1e-4)
        is_dn_open = valid_pre & (ro <= dn_px + 1e-4)

        base_tradable = (tstat == 1) & (rvol > 0) & np.isfinite(ro) & (ro > 0) & np.isfinite(ho) & (ho > 0)
        can_buy[:, s_idx] = base_tradable & (is_st == 0) & (~is_up_open)
        can_sell[:, s_idx] = base_tradable & (~is_dn_open)

        # Sequence channel 0: ret_1d (using forward-filled hfq_close only for active trading days)
        prev_hc = np.roll(hc_ffill.astype(np.float64), 1)
        prev_hc[0] = np.nan
        active_day = (tstat == 1) & (rvol > 0) & np.isfinite(hc) & (hc > 0) & np.isfinite(prev_hc) & (prev_hc > 0)

        ret_1d = np.full(n_days, np.nan, dtype=np.float64)
        ret_1d[active_day] = hc[active_day] / prev_hc[active_day] - 1.0

        # Channel 1: gap_oc = log(hfq_open / prev_hfq_close)
        gap_oc = np.full(n_days, np.nan, dtype=np.float64)
        gap_ok = active_day & np.isfinite(ho) & (ho > 0)
        gap_oc[gap_ok] = np.log(ho[gap_ok] / prev_hc[gap_ok])

        # Channel 2: log_hl = log(raw_high / raw_low)
        log_hl = np.full(n_days, np.nan, dtype=np.float64)
        hl_ok = active_day & np.isfinite(rh) & np.isfinite(rl) & (rh > 0) & (rl > 0) & (rh >= rl)
        log_hl[hl_ok] = np.log(rh[hl_ok] / rl[hl_ok])

        # Channel 3: close_pos = (close - low) / (high - low + eps) - 0.5
        close_pos = np.full(n_days, np.nan, dtype=np.float64)
        cp_ok = hl_ok & np.isfinite(rc)
        close_pos[cp_ok] = (rc[cp_ok] - rl[cp_ok]) / (rh[cp_ok] - rl[cp_ok] + 1e-8) - 0.5

        # Channel 4: log_turn = log1p(max(turn, 0))
        log_turn = np.full(n_days, np.nan, dtype=np.float64)
        turn_ok = active_day & np.isfinite(rturn) & (rturn >= 0)
        log_turn[turn_ok] = np.log1p(rturn[turn_ok])

        # Channel 5: amt_ratio = log((amount + 1) / (MA20(amount) + 1)) strictly trailing
        amt_series = pd.Series(np.where(active_day, ramt, np.nan))
        ma20_amt = amt_series.rolling(window=20, min_periods=5).mean().to_numpy(dtype=np.float64)
        amt_ratio = np.full(n_days, np.nan, dtype=np.float64)
        amt_ok = active_day & np.isfinite(ma20_amt) & (ma20_amt > 0) & (ramt >= 0)
        amt_ratio[amt_ok] = np.log((ramt[amt_ok] + 1.0) / (ma20_amt[amt_ok] + 1.0))

        daily_seq[:, s_idx, 0] = ret_1d.astype(np.float32)
        daily_seq[:, s_idx, 1] = gap_oc.astype(np.float32)
        daily_seq[:, s_idx, 2] = log_hl.astype(np.float32)
        daily_seq[:, s_idx, 3] = close_pos.astype(np.float32)
        daily_seq[:, s_idx, 4] = log_turn.astype(np.float32)
        daily_seq[:, s_idx, 5] = amt_ratio.astype(np.float32)

        # Static PIT features (all strictly <= date t close)
        with np.errstate(divide="ignore", invalid="ignore"):
            ep_ttm = np.where(np.isfinite(pe) & (np.abs(pe) > 1e-4), 1.0 / pe, np.nan)
            bp_mrq = np.where(np.isfinite(pb) & (np.abs(pb) > 1e-4), 1.0 / pb, np.nan)
            roe_ttm = np.where(
                np.isfinite(pb) & (pb > 0.05) & np.isfinite(pe) & (np.abs(pe) > 1e-4),
                pb / pe,
                np.nan,
            )
            log_fmv = np.where(np.isfinite(fmv) & (fmv > 0), np.log(fmv), np.nan)

        roe_series = pd.Series(roe_ttm)
        roe_chg_60d = (roe_series - roe_series.shift(60)).to_numpy(dtype=np.float64)

        hc_series = pd.Series(hc_ffill.astype(np.float64))
        mom_60d = (hc_series / hc_series.shift(60) - 1.0).to_numpy(dtype=np.float64)
        vol_60d = pd.Series(ret_1d).rolling(window=60, min_periods=20).std().to_numpy(dtype=np.float64)

        turn_clean = pd.Series(np.where(turn_ok, rturn, np.nan))
        turn_mean20 = turn_clean.rolling(window=20, min_periods=10).mean()
        turn_std20 = turn_clean.rolling(window=20, min_periods=10).std()
        turn_cv_20d = (turn_std20 / (turn_mean20 + 1e-6)).to_numpy(dtype=np.float64)

        daily_stat[:, s_idx, 0] = ep_ttm.astype(np.float32)
        daily_stat[:, s_idx, 1] = bp_mrq.astype(np.float32)
        daily_stat[:, s_idx, 2] = roe_ttm.astype(np.float32)
        daily_stat[:, s_idx, 3] = roe_chg_60d.astype(np.float32)
        daily_stat[:, s_idx, 4] = log_fmv.astype(np.float32)
        daily_stat[:, s_idx, 5] = mom_60d.astype(np.float32)
        daily_stat[:, s_idx, 6] = vol_60d.astype(np.float32)
        daily_stat[:, s_idx, 7] = turn_cv_20d.astype(np.float32)

    # 4. Assemble weekly cross-sectional tensors with strict cross-sectional normalization
    max_m = max(len([c for c in ml if c in code_to_idx]) for ml in member_lists)
    seq_len = cfg.seq_len
    n_seq_c = len(cfg.seq_features)
    n_stat_c = len(cfg.static_features)

    member_idx = np.full((n_sched, max_m), -1, dtype=np.int32)
    valid_mask = np.zeros((n_sched, max_m), dtype=bool)
    label_valid = np.zeros((n_sched, max_m), dtype=bool)
    X_seq = np.zeros((n_sched, max_m, seq_len, n_seq_c), dtype=np.float32)
    X_stat = np.zeros((n_sched, max_m, n_stat_c), dtype=np.float32)
    X_tab = np.zeros((n_sched, max_m, n_stat_c + 3 * n_seq_c), dtype=np.float32)
    y_raw = np.full((n_sched, max_m), np.nan, dtype=np.float32)
    y_norm = np.zeros((n_sched, max_m), dtype=np.float32)
    y_rank = np.zeros((n_sched, max_m), dtype=np.float32)

    label_end_dates = np.full(n_sched, "", dtype="<U10")
    label_end_cal_idx = np.full(n_sched, -1, dtype=np.int32)

    for k in range(n_sched):
        t_idx = int(signal_cal_idx[k])
        e_start = int(exec_cal_idx[k])
        if k + cfg.horizon_steps < n_sched:
            e_end = int(exec_cal_idx[k + cfg.horizon_steps])
            label_end_cal_idx[k] = e_end
            label_end_dates[k] = exec_dates[k + cfg.horizon_steps]
        else:
            e_end = -1

        s_indices = [code_to_idx[c] for c in member_lists[k] if c in code_to_idx]
        m_cnt = len(s_indices)
        if m_cnt == 0 or t_idx < seq_len:
            continue

        s_arr = np.array(s_indices, dtype=np.int32)
        member_idx[k, :m_cnt] = s_arr

        # Valid if stock has non-NaN close price by signal date t
        has_price = np.isfinite(hfq_close_mtm[t_idx, s_arr])
        valid_mask[k, :m_cnt] = has_price

        # Standardize 20-day sequence cross-sectionally at each step tau in [t-19 .. t]
        raw_win = daily_seq[t_idx - seq_len + 1 : t_idx + 1, s_arr, :]  # (T, m_cnt, C)
        for tau in range(seq_len):
            for c_i in range(n_seq_c):
                X_seq[k, :m_cnt, tau, c_i] = _cs_winsor_zscore(raw_win[tau, :, c_i], has_price)

        # Standardize 6 static features cross-sectionally on signal date t
        raw_st = daily_stat[t_idx, s_arr, :]  # (m_cnt, 6)
        for c_i in range(n_stat_c):
            X_stat[k, :m_cnt, c_i] = _cs_winsor_zscore(raw_st[:, c_i], has_price)

        # Fair tabular features for Ridge / MLP: [static (6), seq_mean (6), seq_std (6), seq_last (6)]
        seq_slice = X_seq[k, :m_cnt, :, :]  # (m_cnt, T, C)
        seq_mean = seq_slice.mean(axis=1)
        seq_std = seq_slice.std(axis=1)
        seq_last = seq_slice[:, -1, :]
        raw_tab = np.concatenate([X_stat[k, :m_cnt, :], seq_mean, seq_std, seq_last], axis=1)
        for j in range(raw_tab.shape[1]):
            X_tab[k, :m_cnt, j] = _cs_winsor_zscore(raw_tab[:, j], has_price)

        # Exact open-to-open horizon label: hfq_open[e_end] / hfq_open[e_start] - 1
        if e_end >= 0:
            p0 = hfq_open[e_start, s_arr].astype(np.float64)
            p1 = hfq_open[e_end, s_arr].astype(np.float64)
            ok = has_price & np.isfinite(p0) & (p0 > 0) & np.isfinite(p1) & (p1 > 0)
            ret_h = np.full(m_cnt, np.nan, dtype=np.float32)
            ret_h[ok] = (p1[ok] / p0[ok] - 1.0).astype(np.float32)
            y_raw[k, :m_cnt] = ret_h
            label_valid[k, :m_cnt] = ok
            y_norm[k, :m_cnt] = _cs_winsor_zscore(ret_h, ok)
            y_rank[k, :m_cnt] = _cs_rank_zscore(ret_h, ok)

    cfg.cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.cache_path,
        signal_dates=signal_dates,
        exec_dates=exec_dates,
        label_end_dates=label_end_dates,
        signal_cal_idx=signal_cal_idx,
        exec_cal_idx=exec_cal_idx,
        label_end_cal_idx=label_end_cal_idx,
        member_idx=member_idx,
        valid_mask=valid_mask,
        label_valid=label_valid,
        X_seq=X_seq,
        X_stat=X_stat,
        X_tab=X_tab,
        y_raw=y_raw,
        y_norm=y_norm,
        y_rank=y_rank,
        cal_dates=cal_dates,
        stock_codes=np.array(available_codes, dtype="<U12"),
        hfq_open=hfq_open,
        hfq_close=hfq_close_mtm,
        can_buy=can_buy,
        can_sell=can_sell,
        bench_h00905=bench_h00905,
        bench_000905=bench_000905,
    )
    return load_panel(cfg)


def load_panel(cfg: ExperimentConfig) -> PanelData:
    """Load precomputed panel cache from disk."""
    data = np.load(cfg.cache_path, allow_pickle=False)
    return PanelData(
        signal_dates=data["signal_dates"],
        exec_dates=data["exec_dates"],
        label_end_dates=data["label_end_dates"],
        signal_cal_idx=data["signal_cal_idx"],
        exec_cal_idx=data["exec_cal_idx"],
        label_end_cal_idx=data["label_end_cal_idx"],
        member_idx=data["member_idx"],
        valid_mask=data["valid_mask"],
        label_valid=data["label_valid"],
        X_seq=data["X_seq"],
        X_stat=data["X_stat"],
        X_tab=data["X_tab"],
        y_raw=data["y_raw"],
        y_norm=data["y_norm"],
        y_rank=data["y_rank"],
        cal_dates=data["cal_dates"],
        stock_codes=data["stock_codes"],
        hfq_open=data["hfq_open"],
        hfq_close=data["hfq_close"],
        can_buy=data["can_buy"],
        can_sell=data["can_sell"],
        bench_h00905=data["bench_h00905"],
        bench_000905=data["bench_000905"],
    )
