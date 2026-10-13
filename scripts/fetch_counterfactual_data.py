"""
scripts/fetch_counterfactual_data.py — Public market-data fetcher for boardroom counterfactuals

Fetches from Bitget public REST via ccxt (NO API keys, NO order endpoints):
  - 5m OHLCV   (per symbol, since --since-ms)
  - 4h OHLCV   (per symbol, for ATR(14) geometry reconstruction)
  - funding rate history (per symbol, 8h settlements)

Cache layout: data/counterfactual/{SYMBOL}_5m.csv, {SYMBOL}_4h.csv, {SYMBOL}_funding.csv
Re-running refreshes the tail (idempotent, dedup on timestamp).

All data is exchange-published market data; every row traces to a live API response.
"""

import os
import sys
import argparse
import time

import pandas as pd
import ccxt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join("data", "counterfactual")

# Symbols observed in boardroom_debates post-cutoff (queried from trading_ledger.db, 2026-10-13)
DEBATE_SYMBOLS = [
    "ADA", "APT", "AVAX", "BNB", "BTC", "ETH", "INJ", "LINK", "NEAR", "SOL", "SUI", "XRP",
]


def _exchange():
    return ccxt.bitget({"options": {"defaultType": "swap"}, "enableRateLimit": True})


def _paginate(fetch_fn, since_ms: int, page_limit: int = 1000) -> pd.DataFrame:
    rows = []
    cursor = since_ms
    while True:
        batch = fetch_fn(cursor)
        if not batch:
            break
        rows.extend(batch)
        last_ts = int(batch[-1][0]) if isinstance(batch[-1], (list, tuple)) else int(batch[-1]["timestamp"])
        if len(batch) < page_limit or last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.25)
    return rows


def fetch_ohlcv(ex, unit: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    sym = f"{unit}/USDT:USDT"
    raw = _paginate(
        lambda c: ex.fetch_ohlcv(sym, timeframe, since=c, limit=1000),
        since_ms,
    )
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_funding(ex, unit: str, since_ms: int) -> pd.DataFrame:
    sym = f"{unit}/USDT:USDT"
    raw = _paginate(
        lambda c: ex.fetch_funding_rate_history(sym, since=c, limit=100),
        since_ms,
        page_limit=100,
    )
    rows = [{"timestamp": int(r["timestamp"]), "funding_rate": float(r["fundingRate"])} for r in raw if r.get("timestamp")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Fetch Bitget public OHLCV + funding for counterfactual replay")
    parser.add_argument("--symbols", type=str, default=",".join(DEBATE_SYMBOLS), help="Comma list of base units, e.g. ADA,BTC")
    parser.add_argument("--since-ms", type=int, default=1790784000000, help="UTC ms start for 5m/funding (default 2026-10-01 00:00 UTC)")
    parser.add_argument("--since-4h-ms", type=int, default=1788192000000, help="UTC ms start for 4h (default 2026-09-01 00:00 UTC; ATR(14) history margin)")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    ex = _exchange()
    units = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for unit in units:
        for tf, suffix, since in (("5m", "_5m", args.since_ms), ("4h", "_4h", args.since_4h_ms)):
            df = fetch_ohlcv(ex, unit, tf, since)
            path = os.path.join(CACHE_DIR, f"{unit}USDT{suffix}.csv")
            save_csv(df, path)
            if not df.empty:
                rng = (pd.to_datetime(df['timestamp'].min(), unit='ms'), pd.to_datetime(df['timestamp'].max(), unit='ms'))
                print(f"[OK] {unit} {tf}: {len(df)} rows {rng[0]} -> {rng[1]}")
            else:
                print(f"[WARN] {unit} {tf}: EMPTY")
        fdf = fetch_funding(ex, unit, args.since_ms)
        save_csv(fdf, os.path.join(CACHE_DIR, f"{unit}USDT_funding.csv"))
        print(f"[OK] {unit} funding: {len(fdf)} rows")

    print(f"[DONE] Cache written under {CACHE_DIR}/")


if __name__ == "__main__":
    main()
