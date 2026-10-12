"""
Sovereign Swarm — Multi-Timeframe Data Downloader
Downloads 1H, 15m, 5m OHLCV data from Binance public API (no API key needed).
Covers last 2 years of data for BTC, ETH, SOL, XRP.
"""
import requests
import pandas as pd
import time
import os
import sys

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

SYMBOLS    = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVALS  = ["1h", "15m", "5m"]
LIMIT      = 1000          # max per Binance request
BASE_URL   = "https://api.binance.com/api/v3/klines"

# 2 years of data per timeframe
CANDLES_NEEDED = {
    "1h":  17520,   # 2yr * 365 * 24
    "15m": 70080,   # 2yr * 365 * 24 * 4
    "5m":  210240,  # 2yr * 365 * 24 * 12
}

def fetch_klines(symbol, interval, total_candles):
    """Fetch historical klines in paginated batches from Binance public API."""
    all_rows = []
    end_time = None
    fetched  = 0
    retries  = 0

    while fetched < total_candles:
        need     = min(LIMIT, total_candles - fetched)
        params   = {"symbol": symbol, "interval": interval, "limit": need}
        if end_time:
            params["endTime"] = end_time

        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            retries += 1
            if retries > 5:
                print(f"  [ERROR] Too many retries for {symbol} {interval}: {e}")
                break
            time.sleep(2)
            continue

        if not data:
            break

        all_rows = data + all_rows          # prepend older data
        end_time = data[0][0] - 1           # go further back in time
        fetched += len(data)
        retries  = 0

        print(f"  {symbol} {interval}: fetched {fetched:,}/{total_candles:,} candles", end="\r", flush=True)
        time.sleep(0.12)                    # stay well under rate limit

        if len(data) < need:               # ran out of history
            break

    print()
    return all_rows

def save_csv(symbol, interval, rows):
    cols = ["timestamp","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base",
            "taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[["timestamp","open","high","low","close","volume"]].copy()
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    sym_clean = symbol.replace("USDT","").lower()
    fname = f"historical_{sym_clean}_usdt_{interval}.csv"
    df.to_csv(fname, index=False)
    print(f"  [SAVED] {fname} — {len(df):,} rows")
    return fname

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BINANCE DATA DOWNLOADER — 1H / 15m / 5m")
    print("  Symbols: BTC, ETH, SOL, XRP | Period: ~2 years")
    print("="*60 + "\n")

    downloaded = []
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            sym_clean = symbol.replace("USDT","").lower()
            existing_1h = f"historical_{sym_clean}_usdt.csv"
            if interval == "1h" and os.path.exists(existing_1h):
                print(f"  [SKIP] {symbol} 1h — using existing {existing_1h}")
                continue

            target = f"historical_{sym_clean}_usdt_{interval}.csv"
            if os.path.exists(target):
                df_check = pd.read_csv(target)
                if len(df_check) > 1000:
                    print(f"  [SKIP] {target} already exists ({len(df_check):,} rows)")
                    continue

            need = CANDLES_NEEDED[interval]
            print(f"\n  Downloading {symbol} {interval} ({need:,} candles)...")
            rows = fetch_klines(symbol, interval, need)
            if rows:
                fname = save_csv(symbol, interval, rows)
                downloaded.append(fname)
            else:
                print(f"  [WARN] No data returned for {symbol} {interval}")

    print(f"\n  Download complete. {len(downloaded)} new files saved.")
    print("  Ready to run multi-timeframe backtest.\n")
