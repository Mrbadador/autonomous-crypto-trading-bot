import os
import sqlite3
import pandas as pd
import logging
import asyncio
from datetime import datetime, timedelta, UTC
from openbb import obb

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MACRO_WARDEN: %(message)s")
log = logging.getLogger("macro_warden")

class MacroWarden:
    """
    V19.0 Macro Financial Warden:
    Ingests institutional traditional finance vectors into the RL Tensor
    utilizing the OpenBB Platform directly.
    """
    def __init__(self, db_path="trading_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes persistence layer for macro intelligence."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_logs (
                        timestamp TEXT,
                        spy_health REAL,
                        dxy_momentum REAL
                    )
                """)
            conn.close()
            log.info(f"Macro intelligence storage online: {self.db_path}")
        except Exception as e:
            log.critical(f"Failed to bootstrap storage: {e}")

    def fetch_macro_vectors(self):
        """Fetches SPY and DXY from OpenBB and computes relative scalar indicators."""
        # Go back 300 days to ensure a healthy 200-period moving average buffer
        start_dt = (datetime.now(UTC) - timedelta(days=350)).strftime("%Y-%m-%d")
        
        try:
            log.info("Executing institutional ingest for SPY (S&P 500)...")
            spy_data = obb.equity.price.historical("SPY", provider="yfinance", start_date=start_dt).to_dataframe()
            
            log.info("Executing institutional ingest for DX-Y.NYB (DXY)...")
            dxy_data = obb.equity.price.historical("DX-Y.NYB", provider="yfinance", start_date=start_dt).to_dataframe()
            
            if spy_data.empty or dxy_data.empty:
                raise ValueError("Received empty datasets from institutional feed.")
            
            # 1. SPY Health Computation (Distance from 200 SMA)
            spy_closes = spy_data['close']
            current_spy = float(spy_closes.iloc[-1])
            sma_200 = float(spy_closes.rolling(window=200).mean().iloc[-1])
            
            # (Price / 200MA) - 1.0 -> scalar % distance
            spy_health = (current_spy / sma_200) - 1.0 if sma_200 > 0 else 0.0
            
            # 2. DXY Momentum Computation (24h / 1-day Rate of Change)
            dxy_closes = dxy_data['close']
            current_dxy = float(dxy_closes.iloc[-1])
            prev_dxy = float(dxy_closes.iloc[-2])
            
            # Daily Pct Change
            dxy_momentum = (current_dxy / prev_dxy) - 1.0 if prev_dxy > 0 else 0.0
            
            log.info(f"[MACRO SNAPSHOT] SPY Distance from 200MA: {spy_health:+.4f} | DXY 1D Delta: {dxy_momentum:+.4f}")
            
            return {
                "spy_health": float(spy_health),
                "dxy_momentum": float(dxy_momentum)
            }
            
        except Exception as e:
            log.error(f"Failed to harvest tradfi data vectors: {e}")
            return None

    def save_vectors(self, metrics):
        """Persists scalar vectors into the centralized data stack."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            with conn:
                # Keep density light, we only care about current state for the active RL tensor
                conn.execute("DELETE FROM macro_logs")
                conn.execute("""
                    INSERT INTO macro_logs (timestamp, spy_health, dxy_momentum)
                    VALUES (?, ?, ?)
                """, (datetime.now(UTC).isoformat(), metrics['spy_health'], metrics['dxy_momentum']))
                conn.commit()
            conn.close()
            log.info("Successfully committed Macro vectors to ledger.")
            return True
        except Exception as e:
            log.error(f"Persistence cycle failure: {e}")
            return False

def run_once():
    warden = MacroWarden()
    metrics = warden.fetch_macro_vectors()
    if metrics:
        warden.save_vectors(metrics)
        print("\n" + "="*50)
        print("         MACRO WARDEN INGEST COMPLETE")
        print("="*50)
        print(f" SPY vs 200MA : {metrics['spy_health']:+.2%}")
        print(f" DXY Momentum : {metrics['dxy_momentum']:+.2%}")
        print("="*50 + "\n")
    else:
        print("WARNING: Macro ingest returned null data.")

if __name__ == "__main__":
    run_once()
