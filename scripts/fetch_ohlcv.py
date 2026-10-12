"""
scripts/fetch_ohlcv.py — Multi-Year Historical OHLCV & Funding Rates Fetcher via CCXT

Downloads >= 1-2 years of authentic 5m, 4H, and 1D OHLCV bars and historical funding rates
for fleet assets (BTC, ETH, SOL, XRP, etc.) and saves to data/historical/<symbol>_<timeframe>.csv.
"""

import os
import sys
import time
import argparse
import ccxt
import pandas as pd
from datetime import datetime, timezone, timedelta

# Asset list (fleet coins)
DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT", "BNB/USDT"
]


def fetch_historical_ohlcv(
    symbol: str,
    timeframe: str = "4h",
    days: int = 730,
    out_dir: str = "data/historical"
) -> str:
    """
    Paginates ccxt.bitget() public fetch_ohlcv to retrieve >= `days` of history.
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "").replace(":USDT", "").upper()
    ccxt_symbol = f"{symbol}:USDT" if not symbol.endswith(":USDT") else symbol

    exchange = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 24 * 3600 * 1000)

    print(f"[*] Fetching {symbol} ({timeframe}) history for past {days} days from Bitget UTA...")
    all_candles = []
    current_since = start_ms

    limit = 200  # Bitget standard per-request limit for historical candles
    max_loops = 2000 if timeframe == "5m" else 500

    loop_count = 0
    while current_since < now_ms and loop_count < max_loops:
        loop_count += 1
        try:
            batch = exchange.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, since=current_since, limit=limit)
            if not batch:
                break

            all_candles.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= current_since:
                # No progress
                break
            current_since = last_ts + 1
            time.sleep(0.08)  # rate limit compliance
        except Exception as e:
            print(f"    [WARN] Batch fetch error at {current_since}: {e}. Retrying after sleep...")
            time.sleep(1.0)
            break

    if not all_candles:
        print(f"    [FAIL] No candles retrieved for {symbol}.")
        return ""

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    out_file = os.path.join(out_dir, f"{clean_sym}_{timeframe}.csv")
    df.to_csv(out_file, index=False)
    print(f"    [OK] Saved {len(df):,} {timeframe} candles to {out_file} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]})")
    return out_file


def fetch_historical_funding_rates(
    symbol: str,
    days: int = 400,
    out_dir: str = "data/historical"
) -> str:
    """
    Paginates Bitget v2 mix market history funding rate endpoint to fetch historical funding rates.
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_sym = symbol.replace("/", "").replace(":USDT", "").upper()
    exchange = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})

    print(f"[*] Fetching {symbol} historical funding rates from Bitget UTA...")
    rates = []
    page = 1
    max_pages = max(1, int(days * 3 / 100) + 2)  # 3 funding events per day (every 8 hours), 100 per page

    while page <= max_pages:
        try:
            r = exchange.publicMixGetV2MixMarketHistoryFundRate({
                "symbol": clean_sym,
                "productType": "USDT-FUTURES",
                "pageSize": "100",
                "pageNo": str(page)
            })
            data = r.get("data", [])
            if not data:
                break
            for d in data:
                rates.append({
                    "timestamp": int(d["fundingTime"]),
                    "funding_rate": float(d["fundingRate"]),
                    "symbol": clean_sym
                })
            page += 1
            time.sleep(0.05)
        except Exception as e:
            print(f"    [WARN] Funding fetch error on page {page} for {symbol}: {e}")
            break

    if not rates:
        print(f"    [FAIL] No funding rates retrieved for {symbol}.")
        return ""

    df_f = pd.DataFrame(rates).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out_file = os.path.join(out_dir, f"{clean_sym}_funding.csv")
    df_f.to_csv(out_file, index=False)
    print(f"    [OK] Saved {len(df_f)} funding records to {out_file} ({df_f['timestamp'].iloc[0]} -> {df_f['timestamp'].iloc[-1]})")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV & funding rates via CCXT")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to fetch")
    parser.add_argument("--timeframes", nargs="+", default=["4h", "1d"], help="Timeframes to fetch")
    parser.add_argument("--days", type=int, default=730, help="Days of history (default 730 = 2 years)")
    parser.add_argument("--fetch-funding", action="store_true", help="Also fetch historical funding rates")
    parser.add_argument("--out-dir", type=str, default="data/historical", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("   CCXT HISTORICAL DATA FETCHER")
    print(f"   Symbols: {args.symbols}")
    print(f"   Timeframes: {args.timeframes} | Window: {args.days} days")
    print(f"   Funding Rates: {args.fetch_funding}")
    print("=" * 70)

    for sym in args.symbols:
        for tf in args.timeframes:
            fetch_historical_ohlcv(sym, timeframe=tf, days=args.days, out_dir=args.out_dir)
        if args.fetch_funding:
            fetch_historical_funding_rates(sym, days=args.days, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
