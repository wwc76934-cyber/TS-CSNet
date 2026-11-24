"""Step 5: Download latest live CSI 500 snapshot data (2026-04-01 .. latest trading day)
via BaoStock for real-market paper trading (模拟交易).

Run this script locally when network access is available:
  D:\\QuantProjects\\regime-factor\\.venv\\Scripts\\python.exe -B D:\\QuantProjects\\tscs-alpha\\scripts\\05_fetch_live_paper_data.py

Estimated runtime: ~2-4 minutes (500 current constituents x ~155 trading days, single-page queries).
Supports automatic resume if interrupted.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import List

import baostock as bs
import baostock.util.socketutil as sock_util
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = PROJECT_ROOT / "data" / "live_202611"
BARS_DIR = LIVE_DIR / "bars"

START_DATE = "2026-04-01"
END_DATE = dt.date.today().strftime("%Y-%m-%d")


class HardenedBaoSession:
    """Hardened BaoStock session that resets broken sockets immediately on timeout/UTF-8 error."""

    def __init__(self, timeout_sec: float = 40.0) -> None:
        self.timeout_sec = timeout_sec
        self._orig_send = sock_util.send_msg
        self._logged_in = False

    def _safe_send(self, msg: str) -> str:
        sock = getattr(sock_util, "sock", None)
        if sock is not None:
            try:
                sock.settimeout(self.timeout_sec)
            except Exception:
                pass
        res = self._orig_send(msg)
        if not res:
            try:
                sock_util.close_socket()
            except Exception:
                pass
            raise ConnectionError("BaoStock returned empty response or timed out")
        return res

    def login(self) -> None:
        sock_util.send_msg = self._safe_send
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")
        self._logged_in = True

    def reconnect(self) -> None:
        try:
            bs.logout()
        except Exception:
            pass
        try:
            sock_util.close_socket()
        except Exception:
            pass
        time.sleep(1.5)
        self.login()

    def logout(self) -> None:
        try:
            bs.logout()
        except Exception:
            pass
        sock_util.send_msg = self._orig_send
        self._logged_in = False


def _rs_to_df(rs) -> pd.DataFrame:
    if rs.error_code != "0":
        raise RuntimeError(f"BaoStock query error {rs.error_code}: {rs.error_msg}")
    rows: List[List[str]] = []
    while rs.next():
        rows.append(rs.get_row_data())
    return pd.DataFrame(rows, columns=rs.fields)


def _fetch_with_retry(sess: HardenedBaoSession, fn, retries: int = 4) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            rs = fn()
            return _rs_to_df(rs)
        except Exception as exc:
            last_err = exc
            print(f"  [WARN] BaoStock query failed (attempt {attempt + 1}/{retries}): {exc}")
            sess.reconnect()
    raise RuntimeError(f"BaoStock query failed after {retries} retries: {last_err}")


def fetch_stock_live(sess: HardenedBaoSession, code: str) -> pd.DataFrame:
    raw_fields = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,isST"
    hfq_fields = "date,code,open,high,low,close"

    df_raw = _fetch_with_retry(
        sess,
        lambda: bs.query_history_k_data_plus(
            code, raw_fields, start_date=START_DATE, end_date=END_DATE, frequency="d", adjustflag="3"
        ),
    )
    df_hfq = _fetch_with_retry(
        sess,
        lambda: bs.query_history_k_data_plus(
            code, hfq_fields, start_date=START_DATE, end_date=END_DATE, frequency="d", adjustflag="1"
        ),
    )
    if df_raw.empty or df_hfq.empty:
        return pd.DataFrame()

    df_hfq = df_hfq.rename(
        columns={"open": "hfq_open", "high": "hfq_high", "low": "hfq_low", "close": "hfq_close"}
    )
    df = df_raw.merge(df_hfq[["date", "hfq_open", "hfq_high", "hfq_low", "hfq_close"]], on="date", how="left")

    num_cols = [
        "open", "high", "low", "close", "preclose", "volume", "amount", "turn",
        "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "isST",
        "hfq_open", "hfq_high", "hfq_low", "hfq_close",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute exact float market cap from amount / (turn / 100) when turn > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        fmv_amt = np.where((df["turn"] > 1e-6) & (df["amount"] > 0), df["amount"] / (df["turn"] / 100.0), np.nan)
        fmv_vol = np.where((df["turn"] > 1e-6) & (df["volume"] > 0) & (df["close"] > 0), df["close"] * df["volume"] / (df["turn"] / 100.0), np.nan)
    df["float_mv"] = np.where(np.isfinite(fmv_amt), fmv_amt, fmv_vol)
    return df.sort_values("date").reset_index(drop=True)


def main() -> None:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    force = "--force" in sys.argv

    sess = HardenedBaoSession(timeout_sec=40.0)
    sess.login()
    t0 = time.time()

    try:
        print(f"[1/4] Fetching latest CSI 500 constituents (up to {END_DATE})...")
        mem_df = _fetch_with_retry(sess, lambda: bs.query_zz500_stocks())
        if mem_df.empty:
            raise RuntimeError("Failed to fetch latest CSI 500 constituents")
        mem_df.to_parquet(LIVE_DIR / "members.parquet", index=False)
        codes = mem_df["code"].astype(str).unique().tolist()
        print(f"  OK: {len(codes)} CSI 500 constituents (updateDate={mem_df['updateDate'].iloc[0]})")

        print("[2/4] Fetching latest industry classification...")
        ind_df = _fetch_with_retry(sess, lambda: bs.query_stock_industry())
        ind_df.to_parquet(LIVE_DIR / "industry.parquet", index=False)
        print(f"  OK: {len(ind_df)} industry rows saved")

        print(f"[3/4] Fetching CSI 500 index (sh.000905) bars {START_DATE}..{END_DATE}...")
        idx_df = _fetch_with_retry(
            sess,
            lambda: bs.query_history_k_data_plus(
                "sh.000905", "date,open,high,low,close,volume,amount,pctChg",
                start_date=START_DATE, end_date=END_DATE, frequency="d", adjustflag="3",
            ),
        )
        for c in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
            idx_df[c] = pd.to_numeric(idx_df[c], errors="coerce")
        idx_df.to_parquet(LIVE_DIR / "index_000905.parquet", index=False)
        latest_date = str(idx_df["date"].max())
        print(f"  OK: {len(idx_df)} index trading days, latest trading date = {latest_date}")

        print(f"[4/4] Fetching recent daily raw+hfq+valuation bars for {len(codes)} stocks...")
        ok_cnt = 0
        skip_cnt = 0
        err_codes: List[str] = []

        for i, code in enumerate(codes, start=1):
            out_p = BARS_DIR / f"{code}.parquet"
            if out_p.exists() and not force:
                existing = pd.read_parquet(out_p)
                if not existing.empty and str(existing["date"].max()) >= latest_date:
                    skip_cnt += 1
                    ok_cnt += 1
                    continue
            try:
                sdf = fetch_stock_live(sess, code)
                if not sdf.empty:
                    sdf.to_parquet(out_p, index=False)
                    ok_cnt += 1
                else:
                    err_codes.append(code)
            except Exception as exc:
                print(f"  [ERR] {code}: {exc}")
                err_codes.append(code)

            if i % 50 == 0 or i == len(codes):
                print(
                    f"  Progress: {i}/{len(codes)} | ok={ok_cnt} (cached={skip_cnt}) | "
                    f"err={len(err_codes)} | elapsed={time.time() - t0:.1f}s"
                )

        print(f"\n=== Live CSI 500 Snapshot Complete ({time.time() - t0:.1f}s) ===")
        print(f"Latest trading date : {latest_date}")
        print(f"Downloaded/cached   : {ok_cnt}/{len(codes)} stocks")
        if err_codes:
            print(f"Missing/error codes : {err_codes}")
    finally:
        sess.logout()


if __name__ == "__main__":
    main()
