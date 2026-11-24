"""Step 6: Train production TS-CSNet (on latest 2023-2025 purged window) and generate
a real-market-compliant 10,000,000 RMB simulated trading portfolio on the latest
live CSI 500 trading day (2026-11).

Outputs:
  - outputs/paper_portfolio_live_full.csv   : Full executable Soft-Tilt portfolio (100-share lots)
  - outputs/paper_portfolio_top20.csv       : Top-20 core holdings with TS-CSNet scores, gates & rationales
  - outputs/paper_portfolio_excluded10.csv  : Top-10 risk-excluded (zeroed) stocks with reasons
  - outputs/paper_portfolio_summary.json    : Capital, cash, holdings count, industry & style profile
  - figures/08_live_paper_portfolio.png     : Visual breakdown of live simulated allocation
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from tscs_alpha.config import ExperimentConfig
from tscs_alpha.dataset import PanelData, _cs_winsor_zscore, load_panel
from tscs_alpha.losses import CompositeICRankLoss
from tscs_alpha.models import TSCSNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 160

LIVE_DIR = PROJECT_ROOT / "data" / "live_202611"
SHARED_INDUSTRY_PATH = Path(r"D:\QuantProjects\dfl-csi-alpha\data\incremental\industry_map.parquet")
INITIAL_CAPITAL = 10_000_000.0  # 1,000 万元模拟增强基金


def _rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 10 or np.std(pred) < 1e-10 or np.std(target) < 1e-10:
        return 0.0
    r = spearmanr(pred, target).statistic
    return 0.0 if not np.isfinite(r) else float(r)


def train_or_load_production_ensemble(panel: PanelData, cfg: ExperimentConfig) -> List[TSCSNet]:
    """Train 3-seed TS-CSNet ensemble on latest 3-year purged window (2023-2025) or load cached weights."""
    weights_path = cfg.output_dir / "live_tscsnet_2025end.pt"
    seq_dim = len(cfg.seq_features)
    stat_dim = len(cfg.static_features)

    if weights_path.exists() and "--retrain" not in sys.argv:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        models = []
        for sd in ckpt["state_dicts"]:
            m = TSCSNet(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, dropout=cfg.dropout)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)
        print(f"Loaded cached production TS-CSNet ensemble ({len(models)} seeds) from {weights_path}")
        return models

    # Latest 3-year purged pool: 2023-01-01 .. 2025-12-31 (labeled)
    sig = np.asarray(panel.signal_dates, dtype="<U10")
    end = np.asarray(panel.label_end_dates, dtype="<U10")
    cand = np.where((sig >= "2023-01-01") & (end != "") & (end <= "2025-12-31") & (panel.label_valid.sum(axis=1) >= 50))[0]
    val_idx = cand[-cfg.val_weeks :]
    first_val_sig = str(sig[val_idx[0]])
    train_idx = np.array([k for k in cand if str(end[k]) <= first_val_sig], dtype=np.int64)

    print(f"Training production 3-seed TS-CSNet ensemble: train={len(train_idx)}w, val={len(val_idx)}w (2023-2025)...")
    torch.set_num_threads(2)

    train_batches = []
    for k in train_idx:
        m = panel.label_valid[k]
        train_batches.append((
            torch.from_numpy(panel.X_seq[k, m]),
            torch.from_numpy(panel.X_stat[k, m]),
            torch.from_numpy(panel.y_rank[k, m]),
        ))
    val_batches = []
    for k in val_idx:
        m = panel.label_valid[k]
        val_batches.append((
            torch.from_numpy(panel.X_seq[k, m]),
            torch.from_numpy(panel.X_stat[k, m]),
            panel.y_raw[k, m],
        ))

    models: List[TSCSNet] = []
    state_dicts = []
    for seed in cfg.seeds:
        t0 = time.time()
        torch.manual_seed(seed + 2026)
        rng = np.random.default_rng(seed + 2026)
        model = TSCSNet(seq_dim=seq_dim, static_dim=stat_dim, hidden_dim=cfg.hidden_dim, num_heads=cfg.num_heads, dropout=cfg.dropout)
        loss_fn = CompositeICRankLoss(ranknet_weight=cfg.ranknet_weight, num_pairs=cfg.ranknet_pairs)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)

        best_ic = -1e9
        best_sd = copy.deepcopy(model.state_dict())
        no_imp = 0
        accum = max(1, cfg.grad_accum_dates)

        for _ep in range(cfg.epochs):
            model.train()
            order = rng.permutation(len(train_batches))
            opt.zero_grad()
            sc = 0
            for pos, idx in enumerate(order):
                xs, xst, yr = train_batches[int(idx)]
                loss = loss_fn(model(xs, xst), yr) / accum
                loss.backward()
                sc += 1
                if sc == accum or pos == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                    opt.zero_grad()
                    sc = 0
            sched.step()

            model.eval()
            vics = []
            with torch.no_grad():
                for xs, xst, yraw in val_batches:
                    vics.append(_rank_ic(model(xs, xst).numpy(), yraw))
            vic = float(np.mean(vics))
            if vic > best_ic + 1e-4:
                best_ic = vic
                best_sd = copy.deepcopy(model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= cfg.patience:
                    break

        model.load_state_dict(best_sd)
        model.eval()
        models.append(model)
        state_dicts.append(best_sd)
        print(f"  Seed {seed}: best val_RankIC={best_ic:+.4f} ({time.time() - t0:.1f}s)")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dicts": state_dicts}, weights_path)
    return models


def _load_industry_map() -> Dict[str, str]:
    imap: Dict[str, str] = {}
    if SHARED_INDUSTRY_PATH.exists():
        sdf = pd.read_parquet(SHARED_INDUSTRY_PATH).sort_values("start_date")
        for code, ind in zip(sdf["code"], sdf["industry"]):
            clean = str(ind).split("_")[-1] if "_" in str(ind) else str(ind)
            imap[str(code)] = clean
    live_ind = LIVE_DIR / "industry.parquet"
    if live_ind.exists():
        ldf = pd.read_parquet(live_ind)
        for code, ind in zip(ldf["code"], ldf["industry"]):
            c = str(code)
            if c not in imap and pd.notna(ind) and str(ind).strip():
                imap[c] = str(ind).strip()
    return imap


def _limit_pct_live(code: str, is_st: int) -> float:
    if is_st == 1:
        return 0.05
    if code.startswith("sh.688") or code.startswith("sz.300") or code.startswith("sz.301"):
        return 0.20
    return 0.10


def _score_with_per_stock_gate(model: TSCSNet, x_seq: torch.Tensor, x_stat: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Run TSCSNet forward pass returning both scores (N,) and per-stock gate mean g_i (N,)."""
    with torch.no_grad():
        _, h_last = model.gru(x_seq)
        h_seq = model.seq_norm(h_last[-1])
        h_stat = model.stat_proj(x_stat)
        gate = torch.sigmoid(model.gate_layer(torch.cat([h_seq, h_stat], dim=-1)))
        h = gate * h_seq + (1.0 - gate) * h_stat
        h_in = model.attn_norm(h).unsqueeze(0)
        attn_out, _ = model.cs_attn(h_in, h_in, h_in, need_weights=False)
        h = h + attn_out.squeeze(0)
        h = h + model.ffn(model.ffn_norm(h))
        scores = model.head(model.head_norm(h)).squeeze(-1).numpy()
        g_stock = gate.mean(dim=-1).numpy()
    return scores, g_stock


def _build_rationale(row: pd.Series) -> str:
    """Generate concise, faithful Chinese rationale strictly from TS-CSNet outputs & PIT metrics."""
    parts = []
    if row["Rank"] <= 50:
        parts.append(f"TS-CSNet 截面打分高居全池第 {int(row['Rank'])}/500 名")
    elif row["Rank"] <= 150:
        parts.append(f"TS-CSNet 截面打分位列全池前 30%（第 {int(row['Rank'])} 名）")
    else:
        parts.append(f"TS-CSNet 评分排名第 {int(row['Rank'])} 名，兼顾中证500市值底仓锚定")

    # Valuation / profitability
    if np.isfinite(row["PE_TTM"]) and 0 < row["PE_TTM"] < 25:
        parts.append(f"低估值安全边际突出（PE_TTM={row['PE_TTM']:.1f}x, PB={row['PB_MRQ']:.2f}x）")
    elif np.isfinite(row["ROE_TTM"]) and row["ROE_TTM"] > 0.10:
        parts.append(f"盈利能力稳健（隐含 ROE_TTM={row['ROE_TTM']*100:.1f}%）")
    elif np.isfinite(row["ROE_Chg_60d"]) and row["ROE_Chg_60d"] > 0.01:
        parts.append(f"近60日盈利趋势改善（ΔROE={row['ROE_Chg_60d']*100:+.1f}%）")

    # Price-volume / turnover behavior
    if np.isfinite(row["Turn_20d"]) and row["Turn_20d"] < 2.0:
        parts.append(f"换手低稳无过热拥挤（20日均换手 {row['Turn_20d']:.2f}%）")
    elif np.isfinite(row["Ret_20d"]) and row["Ret_20d"] < -0.03:
        parts.append(f"20日适度回调具备均值回归修复空间（20日涨跌 {row['Ret_20d']*100:+.1f}%）")
    elif np.isfinite(row["Vol_60d"]) and row["Vol_60d"] < 0.022:
        parts.append(f"波动率受控（60日日波 {row['Vol_60d']*100:.2f}%）")

    # Gate emphasis
    g = row["Gate_Temporal"]
    parts.append(f"门控权重 g={g:.2f}（量价{g*100:.0f}%/基本面{(1-g)*100:.0f}%）")
    return "；".join(parts) + "。"


def main() -> None:
    if not (LIVE_DIR / "members.parquet").exists() or not (LIVE_DIR / "index_000905.parquet").exists():
        raise FileNotFoundError(
            f"Live data not found in {LIVE_DIR}. Please run scripts/05_fetch_live_paper_data.py first."
        )

    cfg = ExperimentConfig()
    panel = load_panel(cfg)
    models = train_or_load_production_ensemble(panel, cfg)

    idx_df = pd.read_parquet(LIVE_DIR / "index_000905.parquet").sort_values("date").reset_index(drop=True)
    cal_dates = idx_df["date"].astype(str).to_numpy()
    latest_date = str(cal_dates[-1])
    n_days = len(cal_dates)
    cal_index = pd.Index(cal_dates, name="date")

    mem_df = pd.read_parquet(LIVE_DIR / "members.parquet")
    code_name_map = dict(zip(mem_df["code"].astype(str), mem_df["code_name"].astype(str)))
    ind_map = _load_industry_map()
    codes = sorted(mem_df["code"].astype(str).unique().tolist())

    bars_dir = LIVE_DIR / "bars"
    valid_codes = [c for c in codes if (bars_dir / f"{c}.parquet").exists()]
    n_stocks = len(valid_codes)
    print(f"Scoring live cross-section on {latest_date}: {n_stocks}/{len(codes)} stocks available...")

    # Compute raw daily sequence (T=20, 6) and static (8) on latest_date strictly matching dataset.py
    seq_len = cfg.seq_len
    raw_seq = np.full((n_stocks, seq_len, 6), np.nan, dtype=np.float32)
    raw_stat = np.full((n_stocks, 8), np.nan, dtype=np.float32)
    meta_rows = []

    for i, code in enumerate(valid_codes):
        df = pd.read_parquet(bars_dir / f"{code}.parquet").drop_duplicates("date").set_index("date").reindex(cal_index)
        ro = df["open"].to_numpy(dtype=np.float64)
        rc = df["close"].to_numpy(dtype=np.float64)
        rh = df["high"].to_numpy(dtype=np.float64)
        rl = df["low"].to_numpy(dtype=np.float64)
        rpre = df["preclose"].to_numpy(dtype=np.float64)
        rvol = df["volume"].fillna(0.0).to_numpy(dtype=np.float64)
        ramt = df["amount"].fillna(0.0).to_numpy(dtype=np.float64)
        rturn = df["turn"].to_numpy(dtype=np.float64)
        tstat = df["tradestatus"].fillna(0).to_numpy(dtype=np.int8)
        is_st = df["isST"].fillna(0).to_numpy(dtype=np.int8)
        ho = df["hfq_open"].to_numpy(dtype=np.float64)
        hc = df["hfq_close"].to_numpy(dtype=np.float64)
        pe = df["peTTM"].to_numpy(dtype=np.float64)
        pb = df["pbMRQ"].to_numpy(dtype=np.float64)
        fmv = df["float_mv"].to_numpy(dtype=np.float64)

        hc_ff = pd.Series(hc).ffill().to_numpy(dtype=np.float64)
        prev_hc = np.roll(hc_ff, 1)
        prev_hc[0] = np.nan
        active = (tstat == 1) & (rvol > 0) & np.isfinite(hc) & (hc > 0) & np.isfinite(prev_hc) & (prev_hc > 0)

        ret_1d = np.where(active, hc / prev_hc - 1.0, np.nan)
        gap_oc = np.where(active & np.isfinite(ho) & (ho > 0), np.log(ho / prev_hc), np.nan)
        hl_ok = active & np.isfinite(rh) & np.isfinite(rl) & (rh > 0) & (rl > 0) & (rh >= rl)
        log_hl = np.where(hl_ok, np.log(rh / rl), np.nan)
        close_pos = np.where(hl_ok & np.isfinite(rc), (rc - rl) / (rh - rl + 1e-8) - 0.5, np.nan)
        turn_ok = active & np.isfinite(rturn) & (rturn >= 0)
        log_turn = np.where(turn_ok, np.log1p(rturn), np.nan)
        ma20_amt = pd.Series(np.where(active, ramt, np.nan)).rolling(20, min_periods=5).mean().to_numpy(dtype=np.float64)
        amt_ratio = np.where(active & np.isfinite(ma20_amt) & (ma20_amt > 0) & (ramt >= 0), np.log((ramt + 1.0) / (ma20_amt + 1.0)), np.nan)

        win = slice(n_days - seq_len, n_days)
        raw_seq[i, :, 0] = ret_1d[win]
        raw_seq[i, :, 1] = gap_oc[win]
        raw_seq[i, :, 2] = log_hl[win]
        raw_seq[i, :, 3] = close_pos[win]
        raw_seq[i, :, 4] = log_turn[win]
        raw_seq[i, :, 5] = amt_ratio[win]

        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.where(np.isfinite(pe) & (np.abs(pe) > 1e-4), 1.0 / pe, np.nan)
            bp = np.where(np.isfinite(pb) & (np.abs(pb) > 1e-4), 1.0 / pb, np.nan)
            roe = np.where(np.isfinite(pb) & (pb > 0.05) & np.isfinite(pe) & (np.abs(pe) > 1e-4), pb / pe, np.nan)
            lfmv = np.where(np.isfinite(fmv) & (fmv > 0), np.log(fmv), np.nan)

        roe_chg = (pd.Series(roe) - pd.Series(roe).shift(60)).to_numpy(dtype=np.float64)
        mom60 = (pd.Series(hc_ff) / pd.Series(hc_ff).shift(60) - 1.0).to_numpy(dtype=np.float64)
        vol60 = pd.Series(ret_1d).rolling(60, min_periods=20).std().to_numpy(dtype=np.float64)
        turn_s = pd.Series(np.where(turn_ok, rturn, np.nan))
        turn_cv = (turn_s.rolling(20, min_periods=10).std() / (turn_s.rolling(20, min_periods=10).mean() + 1e-6)).to_numpy(dtype=np.float64)

        raw_stat[i, :] = [ep[-1], bp[-1], roe[-1], roe_chg[-1], lfmv[-1], mom60[-1], vol60[-1], turn_cv[-1]]

        # Tradability check on latest_date
        last_c = float(rc[-1]) if np.isfinite(rc[-1]) else np.nan
        last_pre = float(rpre[-1]) if np.isfinite(rpre[-1]) else np.nan
        last_st = int(is_st[-1])
        lim = _limit_pct_live(code, last_st)
        is_up_close = np.isfinite(last_c) and np.isfinite(last_pre) and last_pre > 0 and (last_c >= round(last_pre * (1 + lim), 2) - 1e-4)
        buyable = (int(tstat[-1]) == 1) and (rvol[-1] > 0) and (last_st == 0) and np.isfinite(last_c) and (last_c > 0) and (not is_up_close)

        ret20 = float(hc_ff[-1] / hc_ff[-21] - 1.0) if (n_days >= 21 and np.isfinite(hc_ff[-1]) and np.isfinite(hc_ff[-21]) and hc_ff[-21] > 0) else np.nan
        turn20_mean = float(np.nanmean(rturn[-20:])) if np.isfinite(rturn[-20:]).any() else np.nan
        amt20_mean = float(np.nanmean(ramt[-20:])) if np.isfinite(ramt[-20:]).any() else np.nan

        meta_rows.append({
            "Code": code,
            "Name": code_name_map.get(code, ""),
            "Industry": ind_map.get(code, "其他"),
            "Ref_Price": last_c,
            "Float_MV_Yi": float(fmv[-1] / 1e8) if np.isfinite(fmv[-1]) else np.nan,
            "PE_TTM": float(pe[-1]) if np.isfinite(pe[-1]) else np.nan,
            "PB_MRQ": float(pb[-1]) if np.isfinite(pb[-1]) else np.nan,
            "ROE_TTM": float(roe[-1]) if np.isfinite(roe[-1]) else np.nan,
            "ROE_Chg_60d": float(roe_chg[-1]) if np.isfinite(roe_chg[-1]) else np.nan,
            "Ret_20d": ret20,
            "Mom_60d": float(mom60[-1]) if np.isfinite(mom60[-1]) else np.nan,
            "Vol_60d": float(vol60[-1]) if np.isfinite(vol60[-1]) else np.nan,
            "Turn_20d": turn20_mean,
            "Avg_Amt_20d_Wan": amt20_mean / 1e4 if np.isfinite(amt20_mean) else np.nan,
            "Is_ST": last_st,
            "Is_LimitUp_Close": bool(is_up_close),
            "Buyable": bool(buyable),
        })

    valid_mask = np.array([np.isfinite(r["Ref_Price"]) and r["Ref_Price"] > 0 for r in meta_rows], dtype=bool)
    X_seq = np.zeros_like(raw_seq, dtype=np.float32)
    X_stat = np.zeros_like(raw_stat, dtype=np.float32)
    for tau in range(seq_len):
        for c in range(6):
            X_seq[:, tau, c] = _cs_winsor_zscore(raw_seq[:, tau, c], valid_mask)
    for c in range(8):
        X_stat[:, c] = _cs_winsor_zscore(raw_stat[:, c], valid_mask)

    # 3-seed TS-CSNet ensemble inference
    xs_t = torch.from_numpy(X_seq[valid_mask])
    xst_t = torch.from_numpy(X_stat[valid_mask])
    seed_scores, seed_gates = [], []
    for m in models:
        sc, g = _score_with_per_stock_gate(m, xs_t, xst_t)
        sc = (sc - sc.mean()) / (sc.std() + 1e-8)
        seed_scores.append(sc)
        seed_gates.append(g)

    avg_score = np.mean(np.stack(seed_scores, axis=0), axis=0)
    avg_score = (avg_score - avg_score.mean()) / (avg_score.std() + 1e-8)
    avg_gate = np.mean(np.stack(seed_gates, axis=0), axis=0)

    df = pd.DataFrame(meta_rows)
    df["TSCS_Score"] = np.nan
    df["Gate_Temporal"] = np.nan
    df["Z_LogMV"] = X_stat[:, 4]
    df.loc[valid_mask, "TSCS_Score"] = avg_score
    df.loc[valid_mask, "Gate_Temporal"] = avg_gate

    df["Rank"] = df["TSCS_Score"].rank(ascending=False, method="min")

    # Exact Soft-Tilt target weights (identical to backtest: w_base ~ exp(0.6 * z_mv), mult = clip(1 + s, 0, 2.5))
    w_base = np.where(valid_mask, np.exp(0.6 * np.clip(df["Z_LogMV"].to_numpy(), -3.0, 3.0)), 0.0)
    w_base = w_base / (w_base.sum() + 1e-12)
    mult = np.where(valid_mask, np.clip(1.0 + 1.0 * df["TSCS_Score"].fillna(-9.0).to_numpy(), 0.0, 2.5), 0.0)
    w_raw = np.where(df["Buyable"].to_numpy(), w_base * mult, 0.0)
    w_tgt = w_raw / (w_raw.sum() + 1e-12)

    df["Bench_Weight"] = w_base
    df["Target_Weight"] = w_tgt

    # Real-market A-share execution on 10,000,000 RMB initial capital:
    # - Main/ChiNext boards: 100-share integer lots (1手)
    # - STAR board (sh.688*): minimum 200-share initial order (科创板申报下限), then 100-share lots
    # - Stage 1: floor allocation; Stage 2: greedy residual cash sweep to keep cash reserve ~0.5%
    target_cash_reserve = INITIAL_CAPITAL * 0.005  # Keep 0.5% (~50k RMB) liquidity buffer
    investable = (INITIAL_CAPITAL - target_cash_reserve) / (1.0 + cfg.buy_fee_rate)
    shares = np.zeros(len(df), dtype=np.int64)
    tgt_vals = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        if df["Target_Weight"].iloc[i] <= 0 or not df["Buyable"].iloc[i]:
            continue
        code = str(df["Code"].iloc[i])
        px = float(df["Ref_Price"].iloc[i])
        tv = investable * float(df["Target_Weight"].iloc[i])
        tgt_vals[i] = tv
        lots = int(tv // (100.0 * px))
        sh = lots * 100
        # Enforce STAR board (sh.688*) 200-share minimum buy rule
        if code.startswith("sh.688") and 0 < sh < 200:
            sh = 200 if tv >= 150.0 * px else 0
        shares[i] = sh

    # Stage 2: Greedy residual cash sweep sorted by largest under-allocation gap (tgt_val - actual_val)
    spent = float(np.sum(shares * df["Ref_Price"].to_numpy(dtype=np.float64)))
    avail_extra = investable - spent
    for _round in range(3):
        gaps = tgt_vals - shares * df["Ref_Price"].to_numpy(dtype=np.float64)
        order_idx = np.argsort(-gaps)
        added_any = False
        for i in order_idx:
            if not df["Buyable"].iloc[i] or tgt_vals[i] <= 0 or gaps[i] <= 0:
                continue
            code = str(df["Code"].iloc[i])
            px = float(df["Ref_Price"].iloc[i])
            inc_shares = 200 if (code.startswith("sh.688") and shares[i] == 0) else 100
            inc_cost = inc_shares * px
            if inc_cost <= avail_extra:
                shares[i] += inc_shares
                avail_extra -= inc_cost
                added_any = True
        if not added_any:
            break

    df["Shares"] = shares
    df["Market_Value"] = df["Shares"] * df["Ref_Price"]
    est_buy_fee = float(df["Market_Value"].sum() * cfg.buy_fee_rate)
    cash_left = INITIAL_CAPITAL - float(df["Market_Value"].sum()) - est_buy_fee
    df["Actual_Weight"] = df["Market_Value"] / INITIAL_CAPITAL
    df["Active_Weight"] = df["Actual_Weight"] - df["Bench_Weight"]

    held_df = df[df["Shares"] > 0].sort_values("Actual_Weight", ascending=False).reset_index(drop=True)
    held_df["Rationale"] = held_df.apply(_build_rationale, axis=1)

    excl_df = df[(df["Shares"] == 0) & valid_mask].sort_values("TSCS_Score", ascending=True).head(15).reset_index(drop=True)

    # Save CSVs & JSON summary
    held_df.to_csv(cfg.output_dir / "paper_portfolio_live_full.csv", index=False, encoding="utf-8-sig")
    held_df.head(20).to_csv(cfg.output_dir / "paper_portfolio_top20.csv", index=False, encoding="utf-8-sig")
    excl_df.to_csv(cfg.output_dir / "paper_portfolio_excluded15.csv", index=False, encoding="utf-8-sig")

    ind_summary = (
        df[valid_mask]
        .groupby("Industry")[["Actual_Weight", "Bench_Weight", "Active_Weight"]]
        .sum()
        .sort_values("Actual_Weight", ascending=False)
        .reset_index()
    )
    ind_summary.to_csv(cfg.output_dir / "paper_portfolio_industry.csv", index=False, encoding="utf-8-sig")

    summary = {
        "Signal_Date": latest_date,
        "Initial_Capital_RMB": INITIAL_CAPITAL,
        "Invested_Market_Value_RMB": float(df["Market_Value"].sum()),
        "Est_Buy_Commission_Slippage_RMB": est_buy_fee,
        "Cash_Reserve_RMB": cash_left,
        "Cash_Ratio": cash_left / INITIAL_CAPITAL,
        "Num_Universe_Scored": int(valid_mask.sum()),
        "Num_Held_Stocks": int(len(held_df)),
        "Num_Zeroed_Or_Unbuyable": int(valid_mask.sum() - len(held_df)),
        "Num_Unbuyable_Filtered": int((~df["Buyable"] & valid_mask).sum()),
        "Top10_Weight": float(held_df["Actual_Weight"].head(10).sum()),
        "Top20_Weight": float(held_df["Actual_Weight"].head(20).sum()),
        "Max_Single_Weight": float(held_df["Actual_Weight"].max()),
        "Min_Held_Position_RMB": float(held_df["Market_Value"].min()),
        "Mean_Gate_Temporal_All": float(df.loc[valid_mask, "Gate_Temporal"].mean()),
        "Mean_Gate_Temporal_Top20": float(held_df["Gate_Temporal"].head(20).mean()),
        "CSI500_Index_Close": float(idx_df["close"].iloc[-1]),
    }
    with open(cfg.output_dir / "paper_portfolio_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Figure 8: Live Simulated Portfolio Breakdown (Top-12 Industries + Top-15 Core Holdings)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    ax = axes[0]
    top_ind = ind_summary.head(12).iloc[::-1]
    y_i = np.arange(len(top_ind))
    h = 0.36
    ax.barh(y_i + h / 2, top_ind["Actual_Weight"] * 100.0, height=h, color="#1f3b73", label="模拟组合权重 (%)")
    ax.barh(y_i - h / 2, top_ind["Bench_Weight"] * 100.0, height=h, color="#adb5bd", label="中证500基准权重 (%)")
    ax.set_yticks(y_i)
    ax.set_yticklabels(top_ind["Industry"], fontsize=9.2)
    ax.set_xlabel("权重 (%)", fontsize=10.2)
    ax.set_title(f"模拟交易建仓行业配置 Top 12 ({latest_date})", fontsize=11.3, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=8.8)

    ax2 = axes[1]
    top15 = held_df.head(15).iloc[::-1]
    y_s = np.arange(len(top15))
    ax2.barh(y_s, top15["Actual_Weight"] * 100.0, color="#2b5c8f", height=0.62)
    ax2.set_yticks(y_s)
    ax2.set_yticklabels([f"{r['Name']} ({r['Code'][3:]})" for _, r in top15.iterrows()], fontsize=9.0)
    ax2.set_xlabel("实际建仓权重 (%)", fontsize=10.2)
    ax2.set_title("模拟交易前 15 大核心重仓股与 TS-CSNet 排名", fontsize=11.3, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    for yv, (_, r) in zip(y_s, top15.iterrows()):
        ax2.text(r["Actual_Weight"] * 100.0 + 0.02, yv, f"{r['Actual_Weight']*100:.2f}% (#{int(r['Rank'])})", va="center", fontsize=8.2)
    ax2.set_xlim(0, float(top15["Actual_Weight"].max()) * 100.0 + 0.35)

    fig.tight_layout()
    fig.savefig(cfg.figure_dir / "08_live_paper_portfolio.png", bbox_inches="tight")
    plt.close(fig)

    print("\n=== Live Paper Trading Portfolio Generated ===")
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")
    print("\n=== Top 15 Core Holdings ===")
    cols_show = ["Code", "Name", "Industry", "Ref_Price", "Shares", "Market_Value", "Actual_Weight", "Active_Weight", "Rank", "TSCS_Score", "Gate_Temporal"]
    print(held_df[cols_show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
