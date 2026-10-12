# Volatility Warden (V16.4)
# Required: pip install pandas pandas-ta ccxt python-dotenv

import os
import json
import ccxt
import pandas as pd
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv

# 1. Native Indicator Math & Logging Setup
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] VOLATILITY_WARDEN: %(message)s")
log = logging.getLogger("volatility_warden")

# 2. Environment Loading
load_dotenv()
EXCHANGE_ID = os.getenv("EXCHANGE_ID", os.getenv("EXCHANGE", "bitget")).lower()
SYMBOL = os.getenv("SYMBOLS", "BTC/USDT").split(",")[0]

# 3. Data Fetching
def fetch_ohlcv(symbol, timeframe="15m", limit=100):
    try:
        ex_class = getattr(ccxt, EXCHANGE_ID, None) or ccxt.bitget
        ex_opts = {"enableRateLimit": True}
        if EXCHANGE_ID == "bitget":
            ex_opts["options"] = {"defaultType": "swap"}
        exchange = ex_class(ex_opts)
        log.info(f"Fetching {limit} {timeframe} candles for {symbol} from {EXCHANGE_ID}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        log.error(f"Failed to fetch OHLCV: {e}")
        return None

# 4. Indicator Calculation & Logic (Pure Pandas/Numpy)
def analyze_volatility(df):
    if df is None or len(df) < 30:
        return None

    try:
        df = df.copy()
        
        # calculate TR (True Range)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Calculate ATR (Length 14 rolling average)
        df['atr'] = df['tr'].rolling(window=14).mean()
        current_atr = df['atr'].iloc[-1]
        avg_atr_24h = df['atr'].mean()

        # Calculate ADX (Length 14)
        df['up_move'] = df['high'].diff()
        df['down_move'] = -df['low'].diff()
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
        
        tr_smooth = df['tr'].rolling(window=14).sum()
        plus_dm_smooth = df['plus_dm'].rolling(window=14).sum()
        minus_dm_smooth = df['minus_dm'].rolling(window=14).sum()
        
        plus_di = 100.0 * (plus_dm_smooth / tr_smooth)
        minus_di = 100.0 * (minus_dm_smooth / tr_smooth)
        
        dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx_series = dx.rolling(window=14).mean()
        adx = adx_series.iloc[-1]
        
        # Fallbacks for NaN values
        if pd.isna(adx):
            adx = 20.0
        if pd.isna(current_atr):
            current_atr = 0.0
            avg_atr_24h = 1.0

        # Logic Implementation
        market_vibe = "TRENDING"
        risk_mult = 1.0

        # Veto threshold configurations (Tightened: CHOPPY if ADX < 22, was 20)
        if adx < 22.0:
            market_vibe = "CHOPPY"
            risk_mult = 0.5
        elif adx > 25.0:
            market_vibe = "TRENDING"
            risk_mult = 1.0
        else:
            market_vibe = "NEUTRAL"
            risk_mult = 0.75

        # Extreme Check (ATR spike)
        if current_atr > (2 * avg_atr_24h):
            market_vibe = "EXTREME"
            risk_mult = 0.25 

        return {
            "market_vibe": market_vibe,
            "adx_score": round(float(adx), 2),
            "atr_value": round(float(current_atr), 4),
            "recommended_risk_multiplier": risk_mult,
            "last_update": datetime.now().isoformat()
        }
    except Exception as e:
        log.error(f"Analysis Error: {e}. Falling back to NEUTRAL.")
        return {
            "market_vibe": "NEUTRAL",
            "adx_score": 0.0,
            "atr_value": 0.0,
            "recommended_risk_multiplier": 1.0,
            "last_update": datetime.now().isoformat()
        }

# 5. State Persistence
def save_volatility_state(analysis):
    if not analysis:
        return

    state_file = "volatility_state.json"
    try:
        with open(state_file, "w") as f:
            json.dump(analysis, f, indent=4)
        log.info(f"Volatility Warden state saved: Vibe={analysis['market_vibe']}")
    except Exception as e:
        log.error(f"Failed to save volatility state: {e}")

# 6. Main Execution
if __name__ == "__main__":
    log.info("--- Initiating Hardened Volatility Warden Audit ---")
    
    # Step A: Fetch
    df_15m = fetch_ohlcv(SYMBOL, timeframe="15m", limit=50)
    
    # Step B: Analyze
    vol_analysis = analyze_volatility(df_15m)
    
    # Step C: Persist
    save_volatility_state(vol_analysis)
    
    log.info("Volatility Audit Complete.")
