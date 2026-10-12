# analyst_agent.py — MARL Analyst Agent (Feature Engineer)
# Required: pip install numpy

import os
import sqlite3
import json
import numpy as np
import logging
import asyncio

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] ANALYST_AGENT: %(message)s")
log = logging.getLogger("analyst_agent")

def detect_volume_divergence(prices: list, volumes: list) -> dict:
    """
    V22.0 Mathematical Helper: 
    Performs basic linear regression on the recent history slice to detect divergence metrics.
    """
    try:
        if len(prices) < 3 or len(volumes) < 3:
            return {"price_slope": 0.0, "vol_slope": 0.0, "signal": "INSUFFICIENT_DATA"}
        
        # Normalize vectors to compare slopes on relative scale
        p_arr = np.array(prices)
        v_arr = np.array(volumes)
        x = np.arange(len(p_arr))
        
        # Get standardized slopes
        p_slope = float(np.polyfit(x, p_arr / (p_arr[0] or 1.0), 1)[0])
        v_slope = float(np.polyfit(x, v_arr / (v_arr[0] or 1.0), 1)[0])
        
        status = "NEUTRAL"
        if p_slope > 0 and v_slope < 0: status = "BEARISH_DIVERGENCE (Weak Momentum)"
        elif p_slope < 0 and v_slope > 0: status = "ABSORPTION (Potential Markup)"
        elif p_slope < 0 and v_slope < 0: status = "EXHAUSTION (Hidden Bullish Setup)"
        
        return {
            "price_slope": round(p_slope, 4),
            "volume_slope": round(v_slope, 4),
            "divergence_status": status
        }
    except Exception as e:
        log.error(f"Divergence calc failed: {e}")
        return {"divergence_status": "ERROR"}

# --- V22.0 BOARDROOM REASONING DIRECTIVES (PROMPT-ENGINEER GENERATED) ---
BOARDROOM_DIRECTIVES = {
    "BULL_SYSTEM": (
        "Role: You are a high-conviction institutional Market Bull. [RTF - Role]\n"
        "Objective: Argue why this trade entry will expand into a winning breakout. [RODES - Objective]\n"
        "Details:\n"
        "- Prioritize 'Hidden Volume Divergence' as an indicator of supply exhaustion.\n"
        "- Interpret falling volume on price dips as institutional absorption, NOT weakness.\n"
        "Approach step-by-step: [Chain of Thought]\n"
        "1. Examine price/volume correlation supplied in context.\n"
        "2. Validate if 'Hidden Bullish Setup' exists (e.g., price holding while volume dries up).\n"
        "3. Connect current narrative sentiment to your conclusion."
    ),
    
    "BEAR_SYSTEM": (
        "Role: You are a skeptical, hyper-critical Market Bear. [RTF - Role]\n"
        "Objective: Argue that the proposed trade is a trap designed to capture exit liquidity. [RODES - Objective]\n"
        "Details:\n"
        "- Expose 'Weak Momentum Divergence': rising prices on declining volume represent retail FOMO without backing.\n"
        "- Scrutinize volatility and look for exhaustion signatures.\n"
        "Approach step-by-step: [Chain of Thought]\n"
        "1. Evaluate if recent volume growth fails to match price velocity.\n"
        "2. Search context for signs of distributed trap mechanics.\n"
        "3. Vigorously weigh the downside risks of entering now."
    ),
    
    "JUDGE_SYSTEM": (
        "Role: You are a strictly quantitative, unbiased Portfolio Manager. [RTF - Role]\n"
        "Objective: Weigh the Bull and Bear case, focusing heavily on verifying the 'Hidden Volume Divergence' condition, to render an executable JSON verdict.\n"
        "Constraint: Your primary conviction anchor is Volume Integrity.\n"
        "Sense Check: [RODES - Sense check]\n"
        "- If price is ascending but volume is descending, drop conviction by 30%.\n"
        "- If price is consolidating and volume is dropping sharply, raise conviction (Accumulation confirmed).\n"
        "Output: Strict JSON ONLY with 'execute', 'conviction_score' (0.0 - 1.0), and 'final_rationale'."
    )
}

class AnalystAgent:
    """
    Agent 2: The Analyst (Feature Engineer).
    Fetches raw market context and technical indicators, then mathematical-normalizes
    the values into a bounded 1D state tensor (NumPy array) suitable for reinforcement learning.
    """
    def __init__(self, db_path="trading_ledger.db"):
        self.db_path = db_path
        log.info(f"AnalystAgent initialized with DB path: {self.db_path}")

    def _get_db_conn(self, db_path: str, timeout: float = 10.0):
        """Project-standard DB connector with WAL mode and busy timeout."""
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        return conn

    def fetch_market_context(self):
        """
        Connects to the SQLite DB and extracts the latest market conditions.
        Gracefully falls back to mock data if the database or table is missing/incomplete.
        """
        fallback_data = {
            "fear_greed_index": 75.0,
            "volatility_state": 1.2,
            "trend_strength": 0.5,
            "vwap": 50000.0,
            "rsi_14": 50.0
        }
        
        if not os.path.exists(self.db_path):
            log.warning(f"Database {self.db_path} not found. Using fallback market context.")
            return fallback_data

        try:
            with self._get_db_conn(self.db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                
                # V17.0 Institutional Features Extractor (VWAP, RSI)
                cursor.execute("""
                    SELECT fear_greed_index, volatility_state, trend_strength, vwap, rsi_14
                    FROM market_context 
                    WHERE id = 1
                """)
                row = cursor.fetchone()
                
                if row:
                    data = {
                        "fear_greed_index": float(row[0]),
                        "volatility_state": float(row[1]),
                        "trend_strength": float(row[2]),
                        "vwap": float(row[3]) if row[3] is not None else 0.0,
                        "rsi_14": float(row[4]) if row[4] is not None else 50.0
                    }
                    log.info(f"Successfully retrieved raw V17.0 market context: {data}")
                    return data
                else:
                    log.warning("market_context table is empty. Using fallback context.")
                    return fallback_data
                
        except sqlite3.OperationalError as e:
            log.warning(f"Database query failed (likely table missing): {e}. Using fallback context.")
            return fallback_data
        except Exception as e:
            log.error(f"Unexpected database error: {e}. Using fallback context.")
            return fallback_data

    def fetch_technical_data(self, limit=10):
        """
        Fetches live technical execution data.
        Pulls price, ema_ref, and vol_pct from active telemetry tables.
        Returns a dictionary containing list of values of length limit.
        """
        import os
        db_path = "trading_data.db"
        fallback = {
            "ema_distance": [0.02] * limit,
            "volume_multiplier": [1.5] * limit
        }
        if not os.path.exists(db_path):
            return fallback

        try:
            with self._get_db_conn(db_path, timeout=15.0) as conn:
                cursor = conn.cursor()
                
                # Check tables in priority: marl_live_telemetry -> telemetry -> berserker_telemetry
                rows = []
                for table in ["marl_live_telemetry", "telemetry", "berserker_telemetry"]:
                    try:
                        cursor.execute(f"SELECT price, ema_ref, vol_pct FROM {table} ORDER BY timestamp DESC LIMIT ?", (limit,))
                        rows = cursor.fetchall()
                        if rows:
                            # Invert to preserve chronological stream
                            rows = rows[::-1]
                            break
                    except sqlite3.OperationalError:
                        continue
                
                if not rows:
                    return fallback
                
                # If we have fewer than limit rows, pad them using the oldest record
                if len(rows) < limit:
                    pad_size = limit - len(rows)
                    rows = [rows[0]] * pad_size + rows
                
                ema_distances = []
                vol_multipliers = []
                for price, ema_ref, vol_pct in rows:
                    if price is not None and ema_ref is not None and ema_ref > 0:
                        ema_distances.append((price - ema_ref) / ema_ref)
                    else:
                        ema_distances.append(0.02)
                        
                    if vol_pct is not None and vol_pct > 0:
                        vol_multipliers.append(vol_pct / 100.0)
                    else:
                        vol_multipliers.append(1.5)
                        
                return {
                    "ema_distance": ema_distances,
                    "volume_multiplier": vol_multipliers
                }
        except Exception as e:
            log.warning(f"Failed to fetch technical data: {e}")
            return fallback

    async def build_state_tensor(self):
        """
        V30.1 Asynchronous Temporal Brain Update: 
        Reconstructs a pure sequence matrix of shape (10, 15) tracking chronological 
        states back across the last 10 cycles using non-blocking DB threads.
        """
        import pandas as pd
        N = 10 # Sequence Length (Temporal Window)
        
        async def safe_query_async(db, q):
            if not os.path.exists(db): return pd.DataFrame()
            def _run():
                try:
                    with self._get_db_conn(db, timeout=15.0) as conn:
                        # Fetch latest first, then invert to preserve chronological stream
                        df = pd.read_sql_query(q, conn)
                        return df.iloc[::-1].reset_index(drop=True)
                except Exception as e:
                    log.warning(f"Database query failed on {db}: {e}")
                    return pd.DataFrame()
            return await asyncio.to_thread(_run)

        # 1. Vectorized Aggregate Fetch (Asynchronous)
        df_ctx = await safe_query_async(self.db_path, "SELECT fear_greed_index, volatility_state, trend_strength, vwap, rsi_14, signal_readiness FROM market_context ORDER BY rowid DESC LIMIT 10")
        df_cyc = await safe_query_async("trading_data.db", "SELECT price FROM cycles ORDER BY rowid DESC LIMIT 10")
        df_reg = await safe_query_async("trading_data.db", "SELECT hurst, adx FROM market_regime ORDER BY rowid DESC LIMIT 10")
        df_sen = await safe_query_async(self.db_path, "SELECT score FROM sentiment_logs ORDER BY timestamp DESC LIMIT 10")
        df_mac = await safe_query_async(self.db_path, "SELECT spy_health, dxy_momentum FROM macro_logs ORDER BY timestamp DESC LIMIT 10")
        
        # Set alignment length based on primary execution ticker
        L = len(df_ctx) if not df_ctx.empty else 1
        
        # Extension helper to guarantee homogenous lengths across all sparse log tables
        def expand_df(df, columns, target):
            if df.empty:
                return pd.DataFrame({c: [0.0]*target for c in columns})
            if len(df) < target:
                pad = pd.DataFrame([df.iloc[0]] * (target - len(df)), columns=columns)
                df = pd.concat([pad, df], ignore_index=True)
            return df.tail(target).reset_index(drop=True)

        df_ctx = expand_df(df_ctx, ["fear_greed_index", "volatility_state", "trend_strength", "vwap", "rsi_14", "signal_readiness"], L)
        df_cyc = expand_df(df_cyc, ["price"], L)
        df_reg = expand_df(df_reg, ["hurst", "adx"], L)
        df_sen = expand_df(df_sen, ["score"], L)
        df_mac = expand_df(df_mac, ["spy_health", "dxy_momentum"], L)

        # 2. Vectorized Normalization Routine
        n_fg = np.clip((df_ctx["fear_greed_index"].fillna(50.0) - 50.0) / 50.0, -1.0, 1.0).values
        n_vol = np.clip(df_ctx["volatility_state"].fillna(1.0) - 1.0, -1.0, 1.0).values
        n_trend = np.clip(df_ctx["trend_strength"].fillna(0.0), -1.0, 1.0).values
        
        t_data = self.fetch_technical_data(limit=L)
        n_ema = np.clip(np.array(t_data.get("ema_distance", [0.02]*L), dtype=np.float32) * 10.0, -1.0, 1.0)
        n_vol_m = np.clip(np.array(t_data.get("volume_multiplier", [1.5]*L), dtype=np.float32) / 3.0, 0.0, 1.0)
        
        n_hurst = np.clip((df_reg["hurst"].fillna(0.5) - 0.5) * 2.0, -1.0, 1.0).values
        n_adx = np.clip(df_reg["adx"].fillna(20.0) / 100.0, 0.0, 1.0).values
        n_rsi = np.clip((df_ctx["rsi_14"].fillna(50.0) - 50.0) / 50.0, -1.0, 1.0).values
        
        raw_p = df_cyc["price"].fillna(df_ctx["vwap"].fillna(1.0)).values
        raw_v = df_ctx["vwap"].replace(0.0, 1.0).fillna(1.0).values
        n_vwap = np.clip(((raw_p / raw_v) - 1.0) * 10.0, -1.0, 1.0)
        
        n_sen = np.clip(df_sen["score"].fillna(0.0), -1.0, 1.0).values
        n_spy = np.clip(df_mac["spy_health"].fillna(0.0) * 10.0, -1.0, 1.0).values
        n_dxy = np.clip(df_mac["dxy_momentum"].fillna(0.0) * 100.0, -1.0, 1.0).values
        
        # Microstructure OFI Ingestion
        raw_ofi = 0.0
        raw_ofi_velocity = 0.0
        norm_ofi_val = None  # [FIX L4-BUG] Pre-initialize — prevents NameError when JSON exists but timestamp is stale (>90s)
        ofi_file = "microstructure_state.json"
        if os.path.exists(ofi_file):
            for _ in range(5):
                try:
                    with open(ofi_file, "r") as f:
                        ofi_data = json.load(f)
                        import time
                        # [P25-PATCH] Staleness window extended 10s → 90s to survive the 60-second polling cycle
                        if time.time() - ofi_data.get("timestamp", 0) < 90.0:
                            raw_ofi = ofi_data.get("net_ofi_score", 0.0)
                            raw_ofi_velocity = ofi_data.get("ofi_velocity", 0.0)
                            norm_ofi_val = ofi_data.get("normalized_ofi", None)
                    break
                except Exception as e:
                    await asyncio.sleep(0.02)
        
        # Apply adaptive normalizer or fallback tanh squeeze
        n_ofi_scalar = norm_ofi_val if norm_ofi_val is not None else float(np.tanh(raw_ofi / 50.0))
        n_ofi = np.full(L, n_ofi_scalar)
        
        # Phase 16.3: Strategic Optimization
        # Replace redundant regime Hurst/ADX features with microstructure velocity and Overlord signal_readiness
        n_ofi_vel_scalar = float(raw_ofi_velocity) if raw_ofi_velocity is not None else float(np.tanh(raw_ofi_velocity / 10.0))
        n_ofi_vel = np.full(L, n_ofi_vel_scalar)
        n_overlord_readiness = df_ctx["signal_readiness"].fillna(0.50).values
        
        # 3. Assemble Sequential Tensor Structure (15 dimensions)
        matrix = np.column_stack([
            n_fg, n_vol, n_trend, n_ema, n_vol_m,
            n_hurst, n_adx, n_rsi, n_vwap,
            n_sen, n_spy, n_dxy, n_ofi,
            n_ofi_vel, n_overlord_readiness
        ]).astype(np.float32)
        
        # V20.0 Zero-Padding Safety Buffer Prepend (15-dim)
        if matrix.shape[0] < N:
            pad_h = N - matrix.shape[0]
            z_pad = np.zeros((pad_h, 15), dtype=np.float32)
            matrix = np.vstack([z_pad, matrix])
        elif matrix.shape[0] > N:
            matrix = matrix[-N:]
            
        log.info(f"Sequential Tensor Synthesized: shape={matrix.shape}")
        return matrix

if __name__ == "__main__":
    import asyncio
    log.info("--- Starting Analyst Agent Test ---")
    
    async def main():
        # Initialize agent (using default db path)
        agent = AnalystAgent(db_path="trading_ledger.db")
        
        # Run feature engineering pipeline
        state_arr = await agent.build_state_tensor()
        
        # Display details
        print("\n" + "="*50)
        print("           MARL STATE TENSOR OUTPUT")
        print("="*50)
        print(f"NumPy Array: {state_arr}")
        print(f"Data Type:   {state_arr.dtype}")
        print(f"Shape:       {state_arr.shape}")
        print("="*50 + "\n")

    asyncio.run(main())
