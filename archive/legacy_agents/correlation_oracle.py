# correlation_oracle.py — Cross-Asset Correlation Oracle

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
print("          CROSS-ASSET CORRELATION ORACLE BOOTING             ")
print("=" * 65)

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_correlations (
                    timestamp TEXT,
                    lead_asset TEXT,
                    lag_asset TEXT,
                    signal TEXT,
                    beta_score REAL,
                    lag_intensity REAL
                )
            """)
        conn.close()
        print(f"[INIT] Database initialized successfully at '{DB_PATH}' with WAL mode.")
    except Exception as e:
        print(f"[CRITICAL] Database initialization failed: {e}")

def fetch_ohlcv(symbol, interval="1m", limit=35):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            klines = response.json()
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            return df["close"].astype(float).values
    except Exception as e:
        print(f"[ERROR] Failed to fetch {symbol} close prices: {e}")
    return None

def calculate_pearson(x, y):
    if len(x) < 30 or len(y) < 30:
        return 0.0
    # Align to 30 elements
    x_window = x[-30:]
    y_window = y[-30:]
    
    x_mean, y_mean = np.mean(x_window), np.mean(y_window)
    num = np.sum((x_window - x_mean) * (y_window - y_mean))
    den = np.sqrt(np.sum((x_window - x_mean)**2) * np.sum((y_window - y_mean)**2))
    if den == 0:
        return 0.0
    return num / den

def log_correlation(lead_asset, lag_asset, signal, beta, intensity):
    timestamp = datetime.now(UTC).isoformat()
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        with conn:
            conn.execute("""
                INSERT INTO active_correlations (timestamp, lead_asset, lag_asset, signal, beta_score, lag_intensity)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, lead_asset, lag_asset, signal, beta, intensity))
            # [P1-FIX] Prune rows older than 7 days to prevent unbounded growth.
            # At one signal per 15s cycle the table accumulates ~5,760+ rows/day.
            conn.execute(
                "DELETE FROM active_correlations WHERE timestamp < datetime('now', '-7 days')"
            )
        conn.close()
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to log correlation alert: {e}")

def run_correlation_cycle():
    # Fetch 1m closes for BTC and SOL
    btc_closes = fetch_ohlcv("BTCUSDT", interval="1m", limit=35)
    sol_closes = fetch_ohlcv("SOLUSDT", interval="1m", limit=35)
    
    if btc_closes is None or sol_closes is None:
        return
        
    if len(btc_closes) < 30 or len(sol_closes) < 30:
        return
        
    # 1. Pearson Correlation over last 30 minutes
    pearson_corr = calculate_pearson(btc_closes, sol_closes)
    
    # 2. Delta Calculations (last 3 minutes)
    btc_delta = (btc_closes[-1] - btc_closes[-3]) / btc_closes[-3]
    sol_delta = (sol_closes[-1] - sol_closes[-3]) / sol_closes[-3]
    
    # Lag intensity represents the absolute spread of returns
    lag_intensity = abs(btc_delta - sol_delta)
    
    # Only evaluate signals if moving together (Pearson Corr > 0.80)
    if pearson_corr > 0.80:
        # Long condition: BTC violent spike up, SOL lagging behind
        if btc_delta > 0.005 and sol_delta < 0.001:
            print(f"[CORRELATION ORACLE] 🚨 BTC Spike Detected ({btc_delta:+.2%}). SOL Lagging ({sol_delta:+.2%}). Beta: {pearson_corr:.2f}. Logging FLASH LONG to ledger.")
            log_correlation("BTCUSDT", "SOLUSDT", "LONG", pearson_corr, lag_intensity)
            
        # Short condition: BTC violent drop down, SOL lagging behind
        elif btc_delta < -0.005 and sol_delta > -0.001:
            print(f"[CORRELATION ORACLE] 🚨 BTC Dump Detected ({btc_delta:+.2%}). SOL Lagging ({sol_delta:+.2%}). Beta: {pearson_corr:.2f}. Logging FLASH SHORT to ledger.")
            log_correlation("BTCUSDT", "SOLUSDT", "SHORT", pearson_corr, lag_intensity)
            
    else:
        # In case we need to log low correlation state
        pass

init_db()

while True:
    try:
        run_correlation_cycle()
    except Exception as e:
        print(f"[CYCLE ERROR] Correlation cycle exception: {e}")
    time.sleep(15)
