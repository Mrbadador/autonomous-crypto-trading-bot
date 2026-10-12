"""
Fast parallel MTF data downloader — fetches remaining symbols concurrently.
Skips already-downloaded files. Uses threading for 4x speed.
"""
import requests
import pandas as pd
import time
import os
import sys
import threading

os.chdir(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

BASE_URL = "https://api.binance.com/api/v3/klines"
LIMIT    = 1000

CANDLES_NEEDED = {
    "15m": 70080,
    "5m":  210240,
}

# Only what's still missing
TASKS = [
    ("ETHUSDT",  "5m"),
    ("SOLUSDT",  "15m"),
    ("SOLUSDT",  "5m"),
    ("XRPUSDT",  "15m"),
    ("XRPUSDT",  "5m"),
]

lock = threading.Lock()

def log(msg):
    with lock:
        print(msg, flush=True)

def fetch_and_save(symbol, interval):
    sym_clean = symbol.replace("USDT", "").lower()
    fname = f"historical_{sym_clean}_usdt_{interval}.csv"

    if os.path.exists(fname):
        size = os.path.getsize(fname)
        if size > 100_000:
            log(f"  [SKIP] {fname} already exists ({size//1024}KB)")
            return

    total   = CANDLES_NEEDED[interval]
    all_rows = []
    end_time = None
    fetched  = 0
    retries  = 0

    while fetched < total:
        need   = min(LIMIT, total - fetched)
        params = {"symbol": symbol, "interval": interval, "limit": need}
        if end_time:
            params["endTime"] = end_time

        try:
            resp = requests.get(BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            retries += 1
            if retries > 5:
                log(f"  [ERROR] {symbol} {interval}: {e}")
                return
            time.sleep(3)
            continue

        if not data:
            break

        all_rows = data + all_rows
        end_time = data[0][0] - 1
        fetched += len(data)
        retries  = 0

        if fetched % 10000 == 0 or fetched == total:
            log(f"  {symbol} {interval}: {fetched:,}/{total:,}")

        time.sleep(0.15)

        if len(data) < need:
            break

    if not all_rows:
        log(f"  [WARN] No data for {symbol} {interval}")
        return

    cols = ["timestamp","open","high","low","close","volume",
            "close_time","quote_vol","trades","tbb","tbq","ignore"]
    df = pd.DataFrame(all_rows, columns=cols)[["timestamp","open","high","low","close","volume"]].copy()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_csv(fname, index=False)
    log(f"  [SAVED] {fname} — {len(df):,} rows ({os.path.getsize(fname)//1024}KB)")

if __name__ == "__main__":
    print(f"\n  Downloading {len(TASKS)} remaining files in parallel...\n")
    threads = []
    for symbol, interval in TASKS:
        t = threading.Thread(target=fetch_and_save, args=(symbol, interval), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # stagger starts slightly

    for t in threads:
        t.join()

    print("\n  All done. Checking files:")
    for symbol, interval in TASKS:
        sym = symbol.replace("USDT","").lower()
        fname = f"historical_{sym}_usdt_{interval}.csv"
        if os.path.exists(fname):
            print(f"  ✓ {fname} — {os.path.getsize(fname)//1024} KB")
        else:
            print(f"  ✗ {fname} — MISSING")
