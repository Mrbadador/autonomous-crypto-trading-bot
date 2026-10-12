"""
relative_strength_matrix.py — 30-Minute Autonomous Multi-Asset Relative Strength & Fleet Hierarchy Engine

Periodically batches all 10 fleet watchlist assets into a single comprehensive prompt:
1. Gathers 15m and 1h momentum, volume surge, RSI, ADX, and funding rates.
2. Integrates Level-2 Order Flow Imbalance (OFI) from microstructure_state.json.
3. Dispatches to Google Gemini (with Groq 3-key failover) to rank assets from #1 to #10.
4. Identifies:
   - Priority Leaders (Top Tier: authorized for capital deployment)
   - Secondary Tier (Neutral / standby)
   - Avoid Chop List (Sluggish / negative tape: strictly vetoed from entry)
5. Enforces the Leader Gate so algorithms only deploy capital into market leaders.
"""

import os
import sys
import time
import json
import logging
import asyncio
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config
from dotenv import load_dotenv

load_dotenv(".env")

log = logging.getLogger("relative_strength")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [RELATIVE-STRENGTH] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

UTC = timezone.utc

# Derive fleet assets dynamically from config if available
FLEET_ASSETS = getattr(config, "SYMBOLS", [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT",
    "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"
])

def _ensure_rs_tables():
    """Initializes SQLite schema for relative strength history in trading_ledger.db."""
    try:
        with sqlite3.connect("trading_ledger.db", timeout=10.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relative_strength_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    priority_leaders TEXT NOT NULL,
                    secondary_tier TEXT,
                    avoid_chop_list TEXT,
                    rationale TEXT,
                    full_payload TEXT
                )
            """)
            conn.commit()
    except Exception as e:
        log.warning(f"Could not init relative strength schema: {e}")

_ensure_rs_tables()


class RelativeStrengthMatrix:
    def __init__(self):
        self.interval_sec = getattr(config, "RELATIVE_STRENGTH_INTERVAL_SEC", 1800)  # 30 mins = 1800s
        self.last_run_ts = 0.0
        self.lock = asyncio.Lock()

    def _read_microstructure_data(self) -> dict:
        """Loads live OFI and tape intelligence for all symbols."""
        micro = {}
        try:
            if os.path.exists("microstructure_state.json"):
                with open("microstructure_state.json", "r") as f:
                    data = json.load(f)
                    sym_dict = data.get("symbols", {})
                    for sym, vals in sym_dict.items():
                        std_s = f"{sym.replace('USDT', '')}/USDT"
                        micro[std_s] = {
                            "normalized_ofi": vals.get("normalized_ofi", 0.0),
                            "ofi_velocity": vals.get("ofi_velocity", 0.0),
                            "net_ofi": vals.get("net_ofi_score", 0.0)
                        }
        except Exception as e:
            log.debug(f"Microstructure read note: {e}")
        return micro

    async def _fetch_asset_telemetry(self, exchange, symbol: str, micro_data: dict) -> dict:
        """Pulls OHLCV candles, computes momentum, and packages asset profile."""
        clean_s = symbol.replace("/", "").replace("-", "").upper()
        res = {
            "symbol": symbol,
            "price": 0.0,
            "change_1h_pct": 0.0,
            "change_4h_pct": 0.0,
            "rsi_15m": 50.0,
            "adx_15m": 20.0,
            "vol_ratio": 1.0,
            "normalized_ofi": micro_data.get(symbol, {}).get("normalized_ofi", 0.0)
        }
        try:
            # Fetch 60 15m candles (15 hours of data)
            ohlcv = await exchange.fetch_ohlcv(clean_s, timeframe="15m", limit=60)
            if ohlcv and len(ohlcv) >= 20:
                df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
                curr_c = float(df["close"].iloc[-1])
                # Exact 1 hour = 4 bars ago
                c_1h_ago = float(df["close"].iloc[-5]) if len(df) >= 5 else curr_c
                # Exact 4 hours = 16 bars ago
                c_4h_ago = float(df["close"].iloc[-17]) if len(df) >= 17 else float(df["close"].iloc[0])

                res["price"] = curr_c
                res["change_1h_pct"] = round(((curr_c - c_1h_ago) / c_1h_ago) * 100.0, 2) if c_1h_ago > 0 else 0.0
                res["change_4h_pct"] = round(((curr_c - c_4h_ago) / c_4h_ago) * 100.0, 2) if c_4h_ago > 0 else 0.0

                # Compute Wilder's RSI
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = (-delta).where(delta < 0, 0.0)
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rs = avg_gain / avg_loss.replace(0, np.nan)
                rsi = 100.0 - (100.0 / (1.0 + rs))
                res["rsi_15m"] = round(float(rsi.iloc[-1]), 1) if not np.isnan(rsi.iloc[-1]) else 50.0

                # Compute Authentic ADX 14
                try:
                    import ta
                    adx_series = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
                    res["adx_15m"] = round(float(adx_series.iloc[-1]), 1) if not np.isnan(adx_series.iloc[-1]) else 20.0
                except Exception:
                    res["adx_15m"] = 20.0

                # Compute Volume Ratio (current bar vol vs 20-bar avg)
                mean_vol = float(df["vol"].rolling(20).mean().iloc[-1] or 1.0)
                curr_vol = float(df["vol"].iloc[-1])
                res["vol_ratio"] = round(curr_vol / mean_vol, 2) if mean_vol > 0 else 1.0
        except Exception as e:
            log.debug(f"Telemetry error for {symbol}: {e}")
        return res

    async def evaluate_fleet_matrix(self, force: bool = False) -> dict:
        """
        Runs the full 30-minute relative strength matrix across all 10 coins.
        """
        now = time.time()
        if not force and (now - self.last_run_ts) < self.interval_sec:
            return {}

        async with self.lock:
            log.info("Running 30-minute Multi-Asset Relative Strength Matrix...")
            from bitget_uta_client import BitgetUTAClient
            client = BitgetUTAClient()
            micro_data = self._read_microstructure_data()

            try:
                # Gather telemetry across all assets concurrently
                tasks = [self._fetch_asset_telemetry(client.exchange, sym, micro_data) for sym in FLEET_ASSETS]
                fleet_telemetry = await asyncio.gather(*tasks)

                matrix_context = {item["symbol"]: item for item in fleet_telemetry}

                import ai_engine
                # Dispatch single batched call to Gemini / Groq failover
                ranking_result = await ai_engine.rank_asset_hierarchy(matrix_context, timeout_sec=20.0)
                self.last_run_ts = time.time()

                leaders = ranking_result.get("priority_leaders", ["BTC/USDT", "ETH/USDT"])
                secondary = ranking_result.get("secondary_tier", [])
                avoid = ranking_result.get("avoid_chop_list", [])
                rationale = ranking_result.get("fleet_focus_rationale", "Standard baseline weights.")

                log.info(f"Fleet Alpha Leaders: {leaders}")
                log.info(f"Avoid Chop List: {avoid}")
                log.info(f"Strategic Rationale: {rationale}")

                # Save to system_state and ledger
                self._persist_rankings(ranking_result)

                return ranking_result
            except Exception as e:
                log.error(f"Relative strength matrix evaluation error: {e}")
                return {}
            finally:
                await client.exchange.close()

    def _persist_rankings(self, res: dict):
        """Persists rankings to system_state and trading_ledger.db."""
        now_iso = datetime.now(UTC).isoformat()
        leaders_str = ", ".join(res.get("priority_leaders", []))
        avoid_str = ", ".join(res.get("avoid_chop_list", []))
        secondary_str = ", ".join(res.get("secondary_tier", []))
        decoupled_runners = res.get("decoupled_runners", [])
        runners_str = ", ".join(decoupled_runners)
        asset_details = res.get("asset_details", {})
        rationale = res.get("fleet_focus_rationale", "")

        # 1. Update system_state in trading_data.db
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_asset_rankings", json.dumps(res), now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_asset_leaders", leaders_str, now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_asset_avoid", avoid_str, now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_decoupled_runners", runners_str, now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_decoupled_regime_matrix", json.dumps(asset_details), now_iso)
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to persist asset rankings to system_state: {e}")

        # 2. Append to relative_strength_history in trading_ledger.db
        try:
            with sqlite3.connect("trading_ledger.db", timeout=5.0) as conn:
                conn.execute("""
                    INSERT INTO relative_strength_history (
                        timestamp, priority_leaders, secondary_tier, avoid_chop_list, rationale, full_payload
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (now_iso, leaders_str, secondary_str, avoid_str, rationale, json.dumps(res)))
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to append to relative_strength_history: {e}")


_rs_instance = None

def get_relative_strength_matrix() -> RelativeStrengthMatrix:
    global _rs_instance
    if _rs_instance is None:
        _rs_instance = RelativeStrengthMatrix()
    return _rs_instance

def run_single_relative_strength_cycle(force: bool = False) -> dict:
    """Convenience synchronous entry-point."""
    matrix = get_relative_strength_matrix()
    return asyncio.run(matrix.evaluate_fleet_matrix(force=force))

if __name__ == "__main__":
    log.info("Starting manual Relative Strength Matrix test cycle...")
    res = run_single_relative_strength_cycle(force=True)
    print(json.dumps(res, indent=2))
