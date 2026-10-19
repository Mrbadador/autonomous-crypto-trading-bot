import os
import sqlite3
import math
import logging
from datetime import datetime, timedelta, UTC
import yfinance as yf

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MACRO_WARDEN: %(message)s")
log = logging.getLogger("macro_warden")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MacroWarden:
    """
    Macro Financial Warden:
    Ingests institutional traditional finance vectors (SPY, DXY) into the SQLite ledger
    utilizing direct yfinance feeds with finite math guards and append-only logging.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            self.db_path = os.path.join(BASE_DIR, "trading_ledger.db")
        elif not os.path.isabs(db_path):
            self.db_path = os.path.join(BASE_DIR, db_path)
        else:
            self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes persistence layer for macro intelligence."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT DEFAULT 'BTC/USDT',
                        spy_health REAL NOT NULL DEFAULT 0.0,
                        dxy_momentum REAL NOT NULL DEFAULT 0.0,
                        source TEXT DEFAULT 'macro_warden'
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_telemetry_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        spy_price REAL,
                        uup_price REAL,
                        dxy_price REAL,
                        spy_1h_change REAL,
                        uup_1h_change REAL,
                        avg_funding_rate_8h REAL,
                        global_killswitch INTEGER NOT NULL DEFAULT 0,
                        derivatives_skew_halt INTEGER NOT NULL DEFAULT 0,
                        alt_long_pause INTEGER NOT NULL DEFAULT 0,
                        macro_regime TEXT NOT NULL DEFAULT 'UNKNOWN',
                        market_posture TEXT NOT NULL DEFAULT 'TACTICAL_DEFENSE',
                        source TEXT DEFAULT 'macro_warden'
                    )
                """)
            conn.close()
            log.info(f"Macro intelligence storage online: {self.db_path}")
        except Exception as e:
            log.critical(f"Failed to bootstrap storage: {e}")

    def fetch_macro_vectors(self):
        """Fetches SPY and DXY via yfinance with timeout and computes finite scalar indicators."""
        try:
            log.info("Executing institutional ingest for SPY (S&P 500)...")
            spy_ticker = yf.Ticker("SPY")
            spy_data = spy_ticker.history(period="350d", timeout=10.0)
            
            log.info("Executing institutional ingest for DX-Y.NYB (DXY)...")
            dxy_ticker = yf.Ticker("DX-Y.NYB")
            dxy_data = dxy_ticker.history(period="10d", timeout=10.0)
            if dxy_data.empty:
                dxy_ticker = yf.Ticker("UUP")
                dxy_data = dxy_ticker.history(period="10d", timeout=10.0)
            
            if spy_data.empty or dxy_data.empty:
                log.warning("Received empty datasets from Yahoo Finance feed.")
                return None
            
            # 1. SPY Health Computation (Distance from 200 SMA)
            spy_closes = spy_data['Close']
            if len(spy_closes) < 200:
                log.warning(f"Insufficient SPY history ({len(spy_closes)} bars) for 200-SMA.")
                return None
                
            current_spy = float(spy_closes.iloc[-1])
            sma_200 = float(spy_closes.rolling(window=200).mean().iloc[-1])
            
            if not math.isfinite(current_spy) or not math.isfinite(sma_200) or sma_200 <= 0:
                log.warning("Invalid/non-finite SPY closes or 200-SMA.")
                return None
                
            spy_health = (current_spy / sma_200) - 1.0
            if not math.isfinite(spy_health):
                spy_health = 0.0
            
            # 2. DXY Momentum Computation (24h / 1-day Rate of Change)
            dxy_closes = dxy_data['Close']
            if len(dxy_closes) < 2:
                log.warning("Insufficient DXY history bars for momentum calculation.")
                return None
                
            current_dxy = float(dxy_closes.iloc[-1])
            prev_dxy = float(dxy_closes.iloc[-2])
            
            if not math.isfinite(current_dxy) or not math.isfinite(prev_dxy) or prev_dxy <= 0:
                log.warning("Invalid/non-finite DXY closes.")
                return None
                
            dxy_momentum = (current_dxy / prev_dxy) - 1.0
            if not math.isfinite(dxy_momentum):
                dxy_momentum = 0.0
            
            log.info(f"[MACRO SNAPSHOT] SPY Distance from 200MA: {spy_health:+.4f} | DXY 1D Delta: {dxy_momentum:+.4f}")
            
            return {
                "spy_health": float(spy_health),
                "dxy_momentum": float(dxy_momentum),
                "spy_price": current_spy,
                "dxy_price": current_dxy
            }
            
        except Exception as e:
            log.error(f"Failed to harvest tradfi data vectors: {e}")
            return None

    def save_vectors(self, metrics):
        """Append-only persistence into the centralized data stack."""
        if not metrics:
            return False
        spy_health = metrics.get('spy_health', 0.0)
        dxy_momentum = metrics.get('dxy_momentum', 0.0)
        if not math.isfinite(spy_health) or not math.isfinite(dxy_momentum):
            log.error(f"Refusing to save non-finite vectors: spy_health={spy_health}, dxy_momentum={dxy_momentum}")
            return False
            
        now_iso = datetime.now(UTC).isoformat()
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            with conn:
                conn.execute("""
                    INSERT INTO macro_logs (timestamp, symbol, spy_health, dxy_momentum, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (now_iso, 'BTC/USDT', float(spy_health), float(dxy_momentum), 'macro_warden'))
                
                macro_regime_label = 'RISK_ON_BULL' if spy_health > 0.02 and dxy_momentum < 0 else ('DEFENSIVE_BEAR' if spy_health < -0.02 or dxy_momentum > 0.005 else 'NEUTRAL')
                market_posture_label = 'TACTICAL_DEFENSE' if spy_health < 0 else 'BALANCED_CHOP'

                # Also log to append-only macro_telemetry_history
                conn.execute("""
                    INSERT INTO macro_telemetry_history (
                        timestamp, spy_price, dxy_price, macro_regime, market_posture, source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    now_iso,
                    metrics.get('spy_price'),
                    metrics.get('dxy_price'),
                    macro_regime_label,
                    market_posture_label,
                    'macro_warden'
                ))
            conn.close()

            # Wire to trading_data.db system_state so Swing and fleet daemons consume live TradFi macro posture
            trading_data_path = os.path.join(BASE_DIR, "trading_data.db")
            if os.path.exists(trading_data_path):
                try:
                    with sqlite3.connect(trading_data_path, timeout=10.0) as data_conn:
                        data_conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('tradfi_macro_regime', ?, ?)",
                                          (macro_regime_label, now_iso))
                        data_conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_regime', ?, ?)",
                                          (macro_regime_label, now_iso))
                        data_conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('spy_health', ?, ?)",
                                          (str(round(spy_health, 4)), now_iso))
                        data_conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('dxy_momentum', ?, ?)",
                                          (str(round(dxy_momentum, 4)), now_iso))
                        data_conn.commit()
                        log.info(f"[MACRO-WARDEN] Dispatched tradfi_macro_regime & macro_regime '{macro_regime_label}' to trading_data.db system_state.")
                except Exception as de:
                    log.warning(f"[MACRO-WARDEN] system_state broadcast note: {de}")

            log.info("Successfully committed Macro vectors to ledger (append-only).")
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
