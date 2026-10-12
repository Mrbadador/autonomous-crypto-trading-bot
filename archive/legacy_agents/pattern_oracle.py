# pattern_oracle.py — Formation Scout Candlestick & Context Analyzer

import os
import time
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, UTC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "trading_ledger.db")

print("=" * 65)
print("              FORMATION SCOUT ACTIVE DAEMON BOOTING             ")
print("=" * 65)

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_patterns (
                    timestamp TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    pattern_name TEXT,
                    context_score REAL
                )
            """)
        conn.close()
        print(f"[INIT] Database initialized successfully at '{DB_PATH}' with WAL mode.")
    except Exception as e:
        print(f"[CRITICAL] Database initialization failed: {e}")

def fetch_ohlcv(symbol, interval, limit=250):
    clean_sym = symbol.replace("/", "").replace("-", "").upper()
    url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={clean_sym}&granularity={interval}&limit={limit}&productType=USDT-FUTURES"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume", "quote_volume"
                ])
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch Bitget {symbol} {interval} bars: {e}")
    return None

def calculate_ema(series, period=200):
    if len(series) < period:
        return series.mean() if len(series) > 0 else 0.0
    ema = [series.iloc[:period].mean()]
    multiplier = 2.0 / (period + 1.0)
    for price in series.iloc[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])
    return ema[-1]

def calculate_rsi(series, period=14):
    if len(series) < period + 1:
        return 50.0
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * 13 + gain[i]) / 14
        avg_loss = (avg_loss * 13 + loss[i]) / 14
        
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def log_pattern(symbol, timeframe, pattern, score):
    timestamp = datetime.now(UTC).isoformat()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        with conn:
            conn.execute("""
                INSERT INTO active_patterns (timestamp, symbol, timeframe, pattern_name, context_score)
                VALUES (?, ?, ?, ?, ?)
            """, (timestamp, symbol, timeframe, pattern, score))
        conn.close()
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to log pattern: {e}")

def run_scout_cycle():
    symbols = ["BTCUSDT", "SOLUSDT"]
    timeframes = ["5m", "15m", "1h"]
    
    for symbol in symbols:
        # Fetch 1H timeframe for EMA200 Context filter
        df_1h = fetch_ohlcv(symbol, "1h", limit=250)
        ema200_1h = calculate_ema(df_1h["close"], 200) if df_1h is not None else 0.0
        
        for tf in timeframes:
            df = fetch_ohlcv(symbol, tf, limit=100)
            if df is None or len(df) < 5:
                continue
                
            # Current and previous bar variables
            close = df["close"].values
            open_val = df["open"].values
            high = df["high"].values
            low = df["low"].values
            
            i = len(df) - 1 # Current closing candle index
            
            body_prev = abs(close[i-1] - open_val[i-1])
            body_curr = abs(close[i] - open_val[i])
            
            is_bullish_engulfing = (
                close[i-1] < open_val[i-1] and
                close[i] > open_val[i] and
                open_val[i] <= close[i-1] and
                close[i] >= open_val[i-1]
            )
            
            is_bearish_engulfing = (
                close[i-1] > open_val[i-1] and
                close[i] < open_val[i] and
                open_val[i] >= close[i-1] and
                close[i] <= open_val[i-1]
            )
            
            # Hammer (long lower wick)
            lower_wick_ham = min(open_val[i], close[i]) - low[i]
            upper_wick_ham = high[i] - max(open_val[i], close[i])
            is_hammer = (lower_wick_ham >= 2 * body_curr and upper_wick_ham <= body_curr * 0.5)
            
            # Shooting Star (long upper wick)
            upper_wick_ss = high[i] - max(open_val[i], close[i])
            lower_wick_ss = min(open_val[i], close[i]) - low[i]
            is_shooting_star = (upper_wick_ss >= 2 * body_curr and lower_wick_ss <= body_curr * 0.5)
            
            # Context filters
            rsi_tf = calculate_rsi(close, 14)
            peak_20 = np.max(high[-20:])
            near_peak = (close[i] >= peak_20 * 0.99)
            near_ema200 = (abs(low[i] - ema200_1h) / ema200_1h <= 0.005) if ema200_1h > 0 else True
            
            if is_bullish_engulfing and near_ema200:
                print(f"[PATTERN SCOUT] {tf} {symbol}: Bullish Engulfing near EMA200 detected. Logging to ledger.")
                log_pattern(symbol, tf, "Bullish Engulfing", 0.95)
                
            elif is_hammer and near_ema200:
                print(f"[PATTERN SCOUT] {tf} {symbol}: Hammer near EMA200 detected. Logging to ledger.")
                log_pattern(symbol, tf, "Hammer", 0.90)
                
            elif is_shooting_star and (rsi_tf > 70 or near_peak):
                print(f"[PATTERN SCOUT] {tf} {symbol}: Shooting Star near peak/overbought detected. Logging to ledger.")
                log_pattern(symbol, tf, "Shooting Star", 0.92)
                
            elif is_bearish_engulfing and rsi_tf > 65:
                print(f"[PATTERN SCOUT] {tf} {symbol}: Bearish Engulfing in overbought zone detected. Logging to ledger.")
                log_pattern(symbol, tf, "Bearish Engulfing", 0.88)

if __name__ == "__main__":
    init_db()
    while True:
        print(f"--- [PATTERN SCOUT] starting new update cycle: {datetime.now().strftime('%H:%M:%S')} ---")
        try:
            run_scout_cycle()
        except Exception as e:
            print(f"[CYCLE ERROR] Pattern scan cycle exception: {e}")
        print("--- [PATTERN SCOUT] update complete. Sleeping for 5 minutes... ---")
        time.sleep(300)
