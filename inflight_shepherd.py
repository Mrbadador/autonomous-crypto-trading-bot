"""
inflight_shepherd.py — Continuous AI In-Flight Position Shepherding & Runner Architect

Autonomous 15-Minute In-Flight Position Shepherd for Bitget Unified Trading Accounts.
Whenever any position is open across any strategy engine:
1. Feeds live 15m/1h candle structure (ATR, ADX, RSI, trend expansion).
2. Feeds Level 2 Order Book Depth & Order Flow Imbalance (OFI) tape conditions.
3. Decides between:
   - RUNNER_EXPANSION: Strong momentum & absorption; expand target to 3R-5R without choking stop.
   - MOMENTUM_STALL: Momentum waning or heavy ask wall resistance; recommend tighter profit ratchet.
   - HOLDING_NORMAL: Healthy oscillation within expected volatility bounds.
   - HARVEST_PROFIT: Exhaustion point reached at target; take profit.
4. Strictly enforces Rule #173 & #175 Breathing Room:
   - Mutes micro-tightening during first 60 minutes UNLESS price expands >= 1.0x ATR.
5. Safely ratchets stops on Bitget exchange order book if tightening is approved.
6. Dormant (0 calls / $0.00 cost) when in 100% cash shield.
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

log = logging.getLogger("inflight_shepherd")
if not log.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [SHEPHERD] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

UTC = timezone.utc

def _ensure_shepherd_tables():
    """Initializes SQLite schema for in-flight shepherd logs in trading_ledger.db."""
    try:
        with sqlite3.connect("trading_ledger.db", timeout=10.0) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inflight_shepherd_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_roe_pct REAL NOT NULL,
                    r_multiple REAL NOT NULL,
                    holding_mins REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    runner_mode INTEGER NOT NULL DEFAULT 0,
                    tape_condition TEXT,
                    suggested_sl REAL,
                    current_sl REAL,
                    sl_ratcheted INTEGER NOT NULL DEFAULT 0,
                    target_r REAL,
                    diagnosis TEXT,
                    confidence REAL
                )
            """)
            conn.commit()
    except Exception as e:
        log.warning(f"Could not init shepherd schema: {e}")

_ensure_shepherd_tables()


def _verdict_word(decision) -> str:
    """The verdict the AI measured, or the absence token when its payload carried none.

    A response with no "verdict" key — or a blank one — measured nothing, so defaulting it to
    "HOLDING_NORMAL" used to publish an all-clear the AI never gave.
    `inflight_shepherd_logs.verdict` is TEXT NOT NULL, so the absence travels as this non-null token
    — never None — and every consumer renders it as an absence.
    """
    payload = decision if isinstance(decision, dict) else {}
    raw = payload.get("verdict")
    if raw is None or not str(raw).strip():
        return config.VERDICT_ABSENT
    return str(raw).strip().upper()


def _tape_word(decision, tape_data):
    """The L2 tape word the position was measured against, or None when nothing carried one.

    The published default used to be the word "BALANCED" — a tape reading the shepherd had not taken.
    `tape_condition` is a nullable column, so an unmeasured tape publishes None and the consumers say
    "not measured" instead of colouring an invented word.
    """
    src = decision if isinstance(decision, dict) else {}
    packet = tape_data if isinstance(tape_data, dict) else {}
    for candidate in (src.get("tape_condition"), packet.get("tape_condition")):
        if candidate is not None and str(candidate).strip() != "":
            return str(candidate)
    return None


class InflightPositionShepherd:
    """
    Continuous 15-Minute AI In-Flight Position Shepherd.
    Monitors active positions, order book depth, and momentum exhaustion.
    """
    def __init__(self):
        self.last_check_ts = {}  # symbol -> float timestamp
        self.eval_interval_sec = getattr(config, "SHEPHERD_EVAL_INTERVAL_SEC", 900)  # 15 minutes = 900s
        self.lock = asyncio.Lock()

    def _read_microstructure_packet(self, symbol: str) -> dict:
        """Reads real-time L2 OFI, spoofing, and tape dynamics for the asset."""
        clean_sym = symbol.replace("/", "").replace("-", "").upper()
        default_packet = {
            "net_ofi_score": 0.0,
            "normalized_ofi": 0.0,
            "tape_condition": "NEUTRAL_BALANCED",
            "absorption_side": "NONE",
            "spoof_detected": False,
            "headline": "Microstructure neutral."
        }
        try:
            if os.path.exists("microstructure_state.json"):
                with open("microstructure_state.json", "r") as f:
                    m_data = json.load(f)
                    sym_dict = m_data.get("symbols", {}).get(clean_sym, {})
                    tape = m_data.get("tape_intel", {})
                    ghosts = m_data.get("ghost_walls", {})
                    return {
                        "net_ofi_score": sym_dict.get("net_ofi_score", m_data.get("net_ofi_score", 0.0)),
                        "normalized_ofi": sym_dict.get("normalized_ofi", m_data.get("normalized_ofi", 0.0)),
                        "tape_condition": tape.get("tape_condition", "NEUTRAL_BALANCED"),
                        "absorption_side": tape.get("absorption_side", "NONE"),
                        "spoof_detected": ghosts.get("spoof_detected", False),
                        "spoof_side": ghosts.get("spoof_side", "NONE"),
                        "headline": tape.get("headline", "Order flow balanced.")
                    }
        except Exception as e:
            log.debug(f"Microstructure read note: {e}")
        return default_packet

    async def _fetch_candle_metrics(self, exchange, symbol: str) -> dict:
        """Fetches 15m and 1h candles to calculate ATR, ADX, RSI, and moving averages."""
        clean_sym = symbol.replace("/", "").replace("-", "").upper()
        metrics = {
            "atr_15m": 0.0,
            "rsi_15m": 50.0,
            "adx_15m": 20.0,
            "ema20_15m": 0.0,
            "trend_1h": "NEUTRAL",
            "price_advance": 0.0
        }
        try:
            # 15m candles
            ohlcv_15m = await exchange.fetch_ohlcv(clean_sym, timeframe="15m", limit=60)
            if ohlcv_15m and len(ohlcv_15m) >= 20:
                df15 = pd.DataFrame(ohlcv_15m, columns=["ts", "open", "high", "low", "close", "vol"])
                # Compute ATR
                high_low = df15["high"] - df15["low"]
                high_close = (df15["high"] - df15["close"].shift()).abs()
                low_close = (df15["low"] - df15["close"].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                metrics["atr_15m"] = float(tr.rolling(14).mean().iloc[-1])
                
                # Compute RSI
                delta = df15["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                metrics["rsi_15m"] = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

                # Compute EMA20
                metrics["ema20_15m"] = float(df15["close"].ewm(span=20, adjust=False).mean().iloc[-1])

                # Compute ADX approximation
                up_move = df15["high"] - df15["high"].shift()
                down_move = df15["low"].shift() - df15["low"]
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / tr.rolling(14).mean())
                minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / tr.rolling(14).mean())
                dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
                adx = dx.rolling(14).mean().iloc[-1]
                metrics["adx_15m"] = float(adx) if not np.isnan(adx) else 20.0

            # 1h candles for higher-timeframe trend context
            ohlcv_1h = await exchange.fetch_ohlcv(clean_sym, timeframe="1h", limit=30)
            if ohlcv_1h and len(ohlcv_1h) >= 20:
                df1h = pd.DataFrame(ohlcv_1h, columns=["ts", "open", "high", "low", "close", "vol"])
                ema20_1h = df1h["close"].ewm(span=20, adjust=False).mean().iloc[-1]
                last_c = df1h["close"].iloc[-1]
                metrics["trend_1h"] = "BULLISH" if last_c > ema20_1h else "BEARISH"
        except Exception as ce:
            log.debug(f"Candle metric computation note for {symbol}: {ce}")
        return metrics

    async def _query_ai_shepherd(self, pos_context: dict, timeout_sec: float = 12.0) -> dict:
        """
        Queries Primary Google Gemini (or Groq failover pool) for in-flight shepherding.
        """
        import ai_engine

        system_prompt = (
            "You are an Elite Institutional Algorithmic Position Shepherd and Runner Architect. "
            "You oversee an active live cryptocurrency futures position on Bitget UTA. "
            "Your objective is to MAXIMIZE R-multiple on winners while aggressively protecting capital when momentum dies.\n\n"
            "Key Trading Invariants:\n"
            "1. RUNNER_EXPANSION: If the trade is trending strongly (ADX > 25, positive order flow absorption, no ask resistance walls), "
            "classify as RUNNER_EXPANSION, set runner_mode=true, expand target_r to 3.0-5.0R, and keep a generous trailing buffer so it doesn't choke.\n"
            "2. MOMENTUM_STALL: If price has expanded into profit but volume is stalling, RSI is exhausted (>70 for longs), "
            "or heavy ask spoof/absorption walls are blocking progress, classify as MOMENTUM_STALL, set tighten_sl=true with suggested_sl locking in gains.\n"
            "3. HOLDING_NORMAL: Normal healthy oscillation inside expected ATR volatility bounds. Maintain current stop.\n"
            "4. HARVEST_PROFIT: Price has reached extended target (>2.5R to 4R) and momentum is breaking down; lock in profit.\n"
            "5. RULE #173 & #175 BREATHING ROOM: If breathing_room_active=true, DO NOT micro-tighten stop loss unless severe adverse structural failure is detected.\n"
            "6. STRICT STOP CONSTRAINT: For LONG, suggested_sl MUST be strictly HIGHER than current_sl AND strictly LOWER than current_price. Never loosen stops.\n"
            "Return ONLY valid JSON with no markdown."
        )

        user_prompt = (
            f"Active Position Telemetry:\n{json.dumps(pos_context, indent=2)}\n\n"
            "Respond strictly in this JSON format:\n"
            "{\n"
            '  "verdict": "RUNNER_EXPANSION" | "HOLDING_NORMAL" | "MOMENTUM_STALL" | "HARVEST_PROFIT" | "ADVERSE_RISK",\n'
            '  "runner_mode": true,\n'
            '  "tighten_sl": false,\n'
            '  "suggested_sl": float or null,\n'
            '  "target_r": 3.5,\n'
            '  "tape_condition": "BULLISH_ABSORPTION" | "BEARISH_ABSORPTION" | "WALL_RESISTANCE" | "BALANCED",\n'
            '  "diagnosis": "1-2 punchy institutional sentences evaluating momentum, tape, and recommended action",\n'
            '  "confidence": 0.85\n'
            "}"
        )

        # Primary Google Gemini
        client = ai_engine.get_llm("chart_scout")
        client.session = ai_engine._ensure_active_session(client)
        combined = f"{system_prompt}\n\n{user_prompt}"

        data = None
        try:
            data = await ai_engine._query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
        except Exception as ge:
            log.warning(f"Shepherd Gemini error: {ge}. Failing over to Groq...")

        # Failover to Groq
        if not data:
            payload = {
                "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            data = await ai_engine._query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

        if data:
            try:
                raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                res = ai_engine.robust_json_decode(raw)
                if res and isinstance(res, dict):
                    return res
            except Exception as pe:
                log.warning(f"Shepherd JSON parse error: {pe}")

        # No verdict was measured: publish the absence, not a plausible all-clear. The verdict is the
        # absence TOKEN (a non-null string) because inflight_shepherd_logs.verdict is TEXT NOT NULL;
        # tape_condition and confidence are nullable, so they publish None. runner_mode / tighten_sl /
        # suggested_sl stay inert (no action on an unmeasured verdict) and target_r 2.5 is POLICY, not
        # a measurement, so it is untouched.
        return {
            "verdict": config.VERDICT_ABSENT,
            "runner_mode": False,
            "tighten_sl": False,
            "suggested_sl": None,
            "target_r": 2.5,
            "tape_condition": None,
            "diagnosis": "AI shepherd unreachable: no verdict measured. The static algorithmic trailing baseline (policy) is holding.",
            "confidence": None
        }

    async def evaluate_active_positions(self, force: bool = False) -> list:
        """
        Main execution loop. Queries Bitget UTA and local tables for active positions.
        If zero positions open: returns immediately (dormant, 0 cost).
        If positions open: evaluates them against 15-minute schedule.
        """
        async with self.lock:
            from bitget_uta_client import BitgetUTAClient
            client = BitgetUTAClient()
            shepherd_reports = []

            try:
                # 1. Fetch live exchange positions
                positions = await client.fetch_positions()
                active_exchange_pos = [
                    p for p in positions
                    if float(p.get("contracts") or p.get("size") or p.get("total") or 0.0) > 0.0
                ]

                # 2. Sync live position funding telemetry in system_state
                try:
                    f_telemetry = {}
                    now_iso = datetime.now(UTC).isoformat()
                    for p in active_exchange_pos:
                        s_raw = p.get("symbol", "")
                        s_clean = client._normalize_symbol(s_raw)
                        std_s = f"{s_clean.replace('USDT', '')}/USDT"
                        t_fund = float(p.get("totalFunding", 0.0) or 0.0)
                        info = {
                            "symbol": std_s,
                            "raw_symbol": s_raw,
                            "side": str(p.get("posSide", "long")).lower(),
                            "total_funding": t_fund,
                            "open_fee_total": float(p.get("openFeeTotal", 0.0) or 0.0),
                            "close_fee_total": float(p.get("closeFeeTotal", 0.0) or 0.0),
                            "cur_realised_pnl": float(p.get("curRealisedPnl", 0.0) or 0.0),
                            "unrealised_pnl": float(p.get("unrealisedPnl", 0.0) or 0.0),
                            "mark_price": float(p.get("markPrice", 0.0) or 0.0),
                            "avg_price": float(p.get("avgPrice", 0.0) or 0.0),
                            "updated_at": now_iso
                        }
                        f_telemetry[s_raw] = info
                        f_telemetry[std_s] = info
                        f_telemetry[s_raw.upper().replace("/", "").replace("-", "")] = info
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('live_position_funding', ?, ?)",
                                     (json.dumps(f_telemetry), now_iso))
                except Exception as shep_fe:
                    log.debug(f"[SHEPHERD] Note saving live funding telemetry: {shep_fe}")

                # 3. Check local tables for additional context (SL, TP, initial risk, entry_time)
                local_swing_positions = {}
                try:
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                        conn.row_factory = sqlite3.Row
                        # First read open_positions (reconciler fleet table)
                        for r in conn.execute("SELECT * FROM open_positions").fetchall():
                            d = dict(r)
                            if "entry_time" in d and "opened_at" not in d:
                                d["opened_at"] = d["entry_time"]
                            local_swing_positions[d["symbol"]] = d
                            local_swing_positions[d["symbol"].replace("/", "")] = d
                        # Then layer swing_open_positions (if any)
                        for r in conn.execute("SELECT * FROM swing_open_positions").fetchall():
                            d = dict(r)
                            local_swing_positions[d["symbol"]] = d
                            local_swing_positions[d["symbol"].replace("/", "")] = d
                except Exception as e_pos:
                    log.debug(f"[SHEPHERD] Note loading local positions: {e_pos}")

                # If zero open positions across the board
                if not active_exchange_pos and not local_swing_positions:
                    self._save_dormant_status()
                    return []

                log.info(f"[SHEPHERD] Active positions detected: {len(active_exchange_pos)} on Bitget | {len(local_swing_positions)} in Swing DB.")

                now_ts = time.time()
                for p in active_exchange_pos:
                    raw_sym = p.get("symbol", "")
                    clean_sym = client._normalize_symbol(raw_sym)
                    std_sym = f"{clean_sym.replace('USDT', '')}/USDT"

                    # 15-minute pacing per asset
                    last_ts = self.last_check_ts.get(clean_sym, 0.0)
                    time_since = now_ts - last_ts
                    if not force and time_since < self.eval_interval_sec:
                        log.debug(f"[SHEPHERD] {std_sym} evaluated {time_since:.0f}s ago (< {self.eval_interval_sec}s). Skipping.")
                        continue

                    entry_p = float(p.get("entryPrice") or p.get("avgPrice") or 0.0)
                    mark_p = float(p.get("markPrice") or entry_p)
                    size = float(p.get("total") or p.get("contracts") or p.get("size") or 0.0)
                    pos_side = str(p.get("posSide") or p.get("side") or "long").upper()
                    leverage_val = float(p.get("leverage") or 3.0)
                    unrealized_pnl = float(p.get("unrealisedPnl") or 0.0)

                    margin_locked = (entry_p * size) / max(1.0, leverage_val) if entry_p > 0 else 10.0
                    roe_pct = (unrealized_pnl / margin_locked) * 100.0 if margin_locked > 0 else 0.0

                    # Merge with local position metadata if available
                    swing_meta = local_swing_positions.get(std_sym) or local_swing_positions.get(clean_sym) or local_swing_positions.get(raw_sym) or {}
                    current_sl = float(swing_meta.get("stop_loss") or (entry_p * 0.98 if "LONG" in pos_side else entry_p * 1.02))
                    current_tp = float(swing_meta.get("take_profit") or (entry_p * 1.05 if "LONG" in pos_side else entry_p * 0.95))
                    opened_at_str = swing_meta.get("opened_at") or swing_meta.get("entry_time") or datetime.now(UTC).isoformat()

                    try:
                        opened_dt = datetime.fromisoformat(opened_at_str.replace("Z", "+00:00"))
                        holding_mins = max(1.0, (datetime.now(UTC) - opened_dt).total_seconds() / 60.0)
                    except Exception:
                        holding_mins = 60.0

                    # Calculate R-Multiple
                    initial_risk_unit = abs(entry_p - current_sl) if current_sl > 0 else (entry_p * 0.02)
                    price_advance = (mark_p - entry_p) if "LONG" in pos_side else (entry_p - mark_p)
                    r_multiple = round(price_advance / initial_risk_unit, 2) if initial_risk_unit > 0 else 0.0

                    # Fetch Candle Metrics & L2 Microstructure Tape
                    candle_data = await self._fetch_candle_metrics(client.exchange, std_sym)
                    tape_data = self._read_microstructure_packet(std_sym)

                    atr_val = candle_data.get("atr_15m", entry_p * 0.015)
                    has_1_atr_expansion = (price_advance >= atr_val)

                    # Rule #173 & #175 Invariant: Breathing Room check
                    breathing_room_active = (holding_mins < 60.0) and (not has_1_atr_expansion)

                    pos_context = {
                        "symbol": std_sym,
                        "side": pos_side,
                        "entry_price": entry_p,
                        "current_price": mark_p,
                        "unrealized_roe_pct": round(roe_pct, 2),
                        "unrealized_pnl_usd": round(unrealized_pnl, 2),
                        "r_multiple": r_multiple,
                        "holding_duration_mins": round(holding_mins, 1),
                        "current_sl": current_sl,
                        "current_tp": current_tp,
                        "breathing_room_active": breathing_room_active,
                        "atr_15m": round(atr_val, 4),
                        "rsi_15m": round(candle_data.get("rsi_15m", 50.0), 1),
                        "adx_15m": round(candle_data.get("adx_15m", 20.0), 1),
                        "trend_1h": candle_data.get("trend_1h", "NEUTRAL"),
                        "tape_condition": tape_data.get("tape_condition", "NEUTRAL_BALANCED"),
                        "normalized_ofi": tape_data.get("normalized_ofi", 0.0),
                        "spoof_detected": tape_data.get("spoof_detected", False)
                    }

                    log.info(f"[SHEPHERD-SCAN] Evaluating {std_sym} [{pos_side}]: ROE={roe_pct:+.2f}% | R={r_multiple:+.2f}R | Hold={holding_mins:.1f}m | OFI={tape_data['normalized_ofi']:.2f}")

                    # Query AI Shepherd
                    decision = await self._query_ai_shepherd(pos_context, timeout_sec=10.0)
                    self.last_check_ts[clean_sym] = now_ts

                    verdict = _verdict_word(decision)
                    runner_mode = bool(decision.get("runner_mode", False))
                    tighten_sl = bool(decision.get("tighten_sl", False))
                    suggested_sl = decision.get("suggested_sl")
                    diagnosis = str(decision.get("diagnosis", "Monitored."))
                    raw_conf = decision.get("confidence", 0.8)
                    # A published absence is None (the fallback measured no confidence); the column is
                    # nullable, so it travels as None rather than being floated into a number.
                    conf = float(raw_conf) if raw_conf is not None else None
                    target_r = float(decision.get("target_r", 3.0))
                    tape_cond = _tape_word(decision, tape_data)

                    # Feature 3: Climax Exhaustion & Peak Harvest Sentinel
                    # Monitors profitable trades (>= +1.0R or >= +8% ROE or HARVEST_PROFIT verdict)
                    # to detect blow-off spikes, ask walls, or severe volume divergence.
                    if r_multiple >= 1.0 or roe_pct >= 8.0 or verdict == "HARVEST_PROFIT":
                        try:
                            import ai_engine
                            if not hasattr(ai_engine, "detect_climax_exhaustion"):
                                import importlib
                                importlib.reload(ai_engine)
                            climax_ctx = {
                                "symbol": std_sym,
                                "side": pos_side,
                                "entry_price": entry_p,
                                "current_price": mark_p,
                                "unrealized_roe_pct": round(roe_pct, 2),
                                "r_multiple": r_multiple,
                                "holding_mins": round(holding_mins, 1),
                                "rsi_15m": round(candle_data.get("rsi_15m", 50.0), 1),
                                "atr_15m": round(atr_val, 4),
                                "normalized_ofi": tape_data.get("normalized_ofi", 0.0),
                                "tape_condition": tape_data.get("tape_condition", "BALANCED")
                            }
                            climax_res = await ai_engine.detect_climax_exhaustion(climax_ctx, timeout_sec=8.0)
                            c_act = str(climax_res.get("action", "LET_RUNNER_EXPAND")).upper()
                            c_detected = bool(climax_res.get("climax_detected", False))
                            c_reason = climax_res.get("exhaustion_reason", "")
                            c_tight_sl = climax_res.get("tighten_sl_price")

                            if c_detected and c_act == "HARVEST_CLIMAX_NOW":
                                log.warning(f"[CLIMAX-SENTINEL] 🚨 BLOW-OFF CLIMAX EXHAUSTION DETECTED for {std_sym} at {r_multiple:+.2f}R ({roe_pct:+.1f}% ROE)! Reason: {c_reason}")
                                log.info(f"[CLIMAX-HARVEST] Executing instant market close on Bitget for {std_sym} to bank peak gains...")
                                close_res = await client.close_position_order(clean_sym, pos_side, size)
                                verdict = "HARVEST_CLIMAX_EXECUTED"
                                diagnosis = f"Peak Climax Harvest Executed at {r_multiple:+.2f}R: {c_reason}"
                                log.info(f"[CLIMAX-HARVEST] Position {std_sym} banked cleanly! {close_res}")
                            elif c_act == "TIGHTEN_TIGHT_TICK" and c_tight_sl:
                                suggested_sl = float(c_tight_sl)
                                tighten_sl = True
                                verdict = "MOMENTUM_STALL"
                                diagnosis = f"Climax Stop Tighten to ${suggested_sl:,.4f}: {c_reason}"
                                log.info(f"[CLIMAX-SENTINEL] Stop ratchet set to tick level ${suggested_sl:,.4f} for {std_sym}.")
                        except Exception as cl_err:
                            log.warning(f"[CLIMAX-SENTINEL-NOTE] Climax check note: {cl_err}")

                    # Feature 2: In-Flight Early Reversal & Volume Climax Sentinel
                    # Evaluates adverse momentum, order flow divergence, or exhaustion walls before hitting full SL
                    if verdict not in ("HARVEST_CLIMAX_EXECUTED",):
                        try:
                            import ai_engine
                            if not hasattr(ai_engine, "detect_early_reversal"):
                                import importlib
                                importlib.reload(ai_engine)
                            rev_ctx = {
                                "symbol": std_sym,
                                "side": pos_side,
                                "entry_price": entry_p,
                                "current_price": mark_p,
                                "unrealized_roe_pct": round(roe_pct, 2),
                                "r_multiple": r_multiple,
                                "holding_mins": round(holding_mins, 1),
                                "rsi_15m": round(candle_data.get("rsi_15m", 50.0), 1),
                                "atr_15m": round(atr_val, 4),
                                "normalized_ofi": tape_data.get("normalized_ofi", 0.0),
                                "tape_condition": tape_data.get("tape_condition", "BALANCED"),
                                "breathing_room_active": breathing_room_active
                            }
                            rev_res = await ai_engine.detect_early_reversal(rev_ctx, timeout_sec=8.0)
                            if rev_res.get("reversal_warning", False):
                                rev_action = str(rev_res.get("action", "HOLD")).upper()
                                rev_reason = rev_res.get("reasoning", "Adverse divergence detected.")
                                rev_tight_sl = rev_res.get("tighten_sl_price")
                                log.warning(f"[EARLY-REVERSAL] ⚠️ Warning for {std_sym}: Action={rev_action} | Severity={rev_res.get('severity')} | Reason: {rev_reason}")

                                if rev_action == "EXIT_IMMEDIATE" and not breathing_room_active:
                                    log.warning(f"[EARLY-REVERSAL] 🚨 Closing {std_sym} immediately to preserve capital and abort failing trade...")
                                    try:
                                        close_res = await client.close_position_order(clean_sym, pos_side, size)
                                        if close_res:
                                            verdict = "EARLY_REVERSAL_EXIT"
                                            diagnosis = f"Early Reversal Capital Defense Exit: {rev_reason}"
                                        else:
                                            log.error(f"[EARLY-REVERSAL] Close position returned false for {std_sym}. Position retained.")
                                    except Exception as c_err:
                                        log.error(f"[EARLY-REVERSAL] Close position failed on exchange for {std_sym}: {c_err}")
                                elif rev_action == "TIGHTEN_SL" and rev_tight_sl:
                                    suggested_sl = float(rev_tight_sl)
                                    tighten_sl = True
                                    diagnosis = f"Early Reversal Stop Ratchet: {rev_reason}"
                                elif rev_action == "HARVEST_HALF":
                                    half_size = round(size * 0.5, 4)
                                    if half_size > 0:
                                        log.info(f"[DYNAMIC-HARVEST] 🌾 Executing 50% partial profit harvest ({half_size} units) on {std_sym}: {rev_reason}")
                                        try:
                                            await client.close_position_order(clean_sym, pos_side, half_size)
                                        except Exception as h_err:
                                            log.warning(f"[DYNAMIC-HARVEST-ERR] {h_err}")
                                    if "LONG" in pos_side and entry_p > current_sl:
                                        suggested_sl = entry_p + (atr_val * 0.2)
                                        tighten_sl = True
                                        diagnosis = f"Dynamic Harvest 50% + Breakeven Green Ratchet: {rev_reason}"
                                    elif "SHORT" in pos_side and entry_p < current_sl:
                                        suggested_sl = entry_p - (atr_val * 0.2)
                                        tighten_sl = True
                                        diagnosis = f"Dynamic Harvest 50% + Breakeven Green Ratchet: {rev_reason}"
                        except Exception as rev_err:
                            log.warning(f"[EARLY-REVERSAL-NOTE] {std_sym}: {rev_err}")

                    # Feature 5: Pre-Funding Settlement Carry Optimization (Funding Bleed Defense)
                    # Inspects position within 45 minutes of 8h settlement (00:00, 08:00, 16:00 UTC)
                    if verdict not in ("HARVEST_CLIMAX_EXECUTED", "EARLY_REVERSAL_EXIT"):
                        try:
                            import ai_engine
                            if not hasattr(ai_engine, "evaluate_funding_carry_risk"):
                                import importlib
                                importlib.reload(ai_engine)
                            now_utc = datetime.now(UTC)
                            curr_hour_dec = now_utc.hour + now_utc.minute / 60.0 + now_utc.second / 3600.0
                            next_settle_hour = min([h for h in [0, 8, 16, 24] if h > curr_hour_dec], default=24)
                            mins_to_settle = (next_settle_hour - curr_hour_dec) * 60.0

                            if mins_to_settle <= 45.0:
                                fr_val = 0.0001
                                try:
                                    fr_meta = await client.exchange.fetch_funding_rate(clean_sym + ":USDT")
                                    if fr_meta and "fundingRate" in fr_meta and fr_meta["fundingRate"] is not None:
                                        fr_val = float(fr_meta["fundingRate"])
                                except Exception:
                                    pass

                                notional_usd = entry_p * size
                                proj_funding_usd = notional_usd * fr_val if "LONG" in pos_side else -(notional_usd * fr_val)

                                carry_ctx = {
                                    "symbol": std_sym,
                                    "side": pos_side,
                                    "entry_price": entry_p,
                                    "current_price": mark_p,
                                    "unrealized_pnl_usd": round(unrealized_pnl, 2),
                                    "r_multiple": r_multiple,
                                    "holding_mins": round(holding_mins, 1),
                                    "funding_rate_8h": fr_val,
                                    "projected_funding_payment_usd": round(proj_funding_usd, 4),
                                    "minutes_to_settlement": round(mins_to_settle, 1)
                                }
                                carry_res = await ai_engine.evaluate_funding_carry_risk(carry_ctx, timeout_sec=8.0)
                                c_risk = str(carry_res.get("carry_risk", "LOW")).upper()
                                c_act = str(carry_res.get("action", "HOLD")).upper()
                                c_reason = carry_res.get("reasoning", "")

                                if c_risk == "HIGH" and c_act == "HARVEST_BEFORE_SETTLEMENT" and holding_mins >= 60.0:
                                    log.warning(f"[CARRY-OPTIMIZER] 💸 HIGH FUNDING DRAG for {std_sym} ({fr_val*100:+.4f}% in {mins_to_settle:.0f}m). Harvesting before settlement...")
                                    close_res = await client.close_position_order(clean_sym, pos_side, size)
                                    verdict = "FUNDING_CARRY_HARVEST"
                                    diagnosis = f"Funding Carry Pre-Settlement Harvest ({fr_val*100:+.4f}%): {c_reason}"
                                elif c_act == "TIGHTEN_TO_BREAKEVEN":
                                    if "LONG" in pos_side and entry_p > current_sl:
                                        suggested_sl = entry_p
                                        tighten_sl = True
                                        diagnosis = f"Funding Carry Breakeven Shield: {c_reason}"
                        except Exception as carry_err:
                            log.warning(f"[CARRY-OPTIMIZER-NOTE] {std_sym}: {carry_err}")

                    # Feature 6: Bayesian Markov Lookahead Breakdown Shield
                    # FIX 2: Log-only until input calibrated; measured baseline min is 0.2588 (prior baseline min was 0.7141),
                    # so breakdown_risk_4step >= 0.60 is over-sensitive (88.5% pass rate). tighten_sl is held False.
                    if r_multiple >= 1.0 and verdict not in ("HARVEST_CLIMAX_EXECUTED", "EARLY_REVERSAL_EXIT"):
                        try:
                            def _read_markov_risk():
                                try:
                                    max_age = float(getattr(config, "REGIME_MAX_AGE_SEC", 900.0) or 900.0)
                                except (TypeError, ValueError):
                                    max_age = 900.0
                                try:
                                    with sqlite3.connect("trading_data.db", timeout=3.0) as conn:
                                        row = conn.execute("SELECT value, updated_at FROM system_state WHERE key = 'markov_regime_telemetry'").fetchone()
                                        if row and row[0]:
                                            raw_val, raw_ts = row[0], row[1]
                                            age = None
                                            if raw_ts:
                                                try:
                                                    cleaned = str(raw_ts).strip().replace("Z", "+00:00")
                                                    if "T" not in cleaned and " " in cleaned:
                                                        cleaned = cleaned.replace(" ", "T")
                                                    parsed_ts = datetime.fromisoformat(cleaned)
                                                    if parsed_ts.tzinfo is None:
                                                        parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
                                                    age = max(0.0, time.time() - parsed_ts.timestamp())
                                                except Exception:
                                                    age = None
                                            if age is not None and age > max_age:
                                                log.warning(f"[SHEPHERD-MARKOV-STALE] ⚠️ markov_regime_telemetry is stale ({age:.1f}s > {max_age:.1f}s). Skipping Markov risk shield.")
                                                return {}
                                            parsed = json.loads(raw_val)
                                            if "is_tradable" not in parsed:
                                                age_str = f"{age:.1f}s" if age is not None else "unknown"
                                                log.warning(f"[SHEPHERD-MARKOV-KEY-MISSING] ⚠️ markov_regime_telemetry missing 'is_tradable' key (age: {age_str}).")
                                            if "sizing_multiplier" not in parsed:
                                                age_str = f"{age:.1f}s" if age is not None else "unknown"
                                                log.warning(f"[SHEPHERD-MARKOV-KEY-MISSING] ⚠️ markov_regime_telemetry missing 'sizing_multiplier' key (age: {age_str}).")
                                            return parsed
                                except Exception:
                                    pass
                                return {}
                            m_tel = await asyncio.to_thread(_read_markov_risk)
                            if m_tel:
                                bd_risk = float(m_tel.get("breakdown_risk_4step", 0.0) or 0.0)
                                if bd_risk >= 0.60:
                                    log.info(f"[MARKOV-SHEPHERD-SHIELD-LOG-ONLY] 🛡️ 4-step breakdown risk ({bd_risk:.2f} >= 0.60) detected for {std_sym} at {r_multiple:+.2f}R (advisory log-only; ratchet inhibited).")
                                    tighten_sl = False
                        except Exception as m_err:
                            log.debug(f"Markov shepherd check note: {m_err}")

                    sl_ratcheted = False
                    # Execute SL Ratchet if approved and valid
                    if tighten_sl and suggested_sl is not None and verdict not in ("HARVEST_CLIMAX_EXECUTED", "EARLY_REVERSAL_EXIT", "FUNDING_CARRY_HARVEST"):
                        suggested_sl = float(suggested_sl)
                        is_valid_ratchet = False

                        if "LONG" in pos_side:
                            # Long invariant: tighter means strictly higher than current_sl and below mark_p
                            if suggested_sl > current_sl and suggested_sl < mark_p:
                                is_valid_ratchet = True
                        else:
                            # Short invariant: tighter means strictly lower than current_sl and above mark_p
                            if suggested_sl < current_sl and suggested_sl > mark_p:
                                is_valid_ratchet = True

                        if is_valid_ratchet:
                            if breathing_room_active and r_multiple < 0.8:
                                log.info(f"[BREATHING-ROOM] Rule #173/175 Active: Muting micro-tightening for {std_sym} (Hold: {holding_mins:.1f}m < 60m & advance < 1.0 ATR). Allowing room to expand.")
                            else:
                                # Coordinate with swing_architect to prevent racing modification on the same position
                                def _is_swing_locked():
                                    try:
                                        with sqlite3.connect("trading_data.db", timeout=5.0) as l_conn:
                                            l_row = l_conn.execute("SELECT locked_by, heartbeat FROM asset_locks WHERE symbol = ?", (std_sym,)).fetchone()
                                            # Fail-closed: if unrecorded or owned by an active fleet engine, treat as locked.
                                            # Only return False (allow external shepherd ratchet) if explicitly unlocked or owned by 'external_shepherd'.
                                            if not l_row:
                                                return True
                                            holder = str(l_row[0] or "")
                                            if holder in ("swing_architect", "super_bot_grid", "super_bot_scalp_short"):
                                                return True
                                            hb_str = str(l_row[1] or "")
                                            clean_hb = hb_str.replace("Z", "+00:00")
                                            if "+" not in clean_hb:
                                                clean_hb += "+00:00"
                                            hb_dt = datetime.fromisoformat(clean_hb)
                                            # Fresh lock only if heartbeat within last 900 seconds (15 mins)
                                            return (datetime.now(timezone.utc) - hb_dt).total_seconds() <= 900.0
                                    except Exception as db_e:
                                        # Fail-closed on DB contention/lock: do not allow shepherd to modify stops blindly
                                        log.warning(f"[SHEPHERD-FAIL-CLOSED] Error reading asset_locks for {std_sym}: {db_e}. Assuming locked.")
                                        return True

                                swing_is_owner = await asyncio.to_thread(_is_swing_locked)
                                if swing_is_owner:
                                    log.info(f"[SHEPHERD-COORDINATION] {std_sym} is actively managed by swing_architect. Inflight Shepherd acting in advisory mode.")
                                    diagnosis = f"Advisory Shepherding: {verdict}"
                                else:
                                    log.info(f"[SHEPHERD-RATCHET] {std_sym} SL Ratchet Approved! Moving SL from ${current_sl:,.4f} -> ${suggested_sl:,.4f} ({verdict})")
                                    try:
                                        res = await client.modify_strategy_order(std_sym, sl=suggested_sl, tp=current_tp)
                                        if res:
                                            sl_ratcheted = True
                                            # Only update local swing state if exchange confirms modification
                                            self._update_local_sl(std_sym, suggested_sl)
                                            log.info(f"[SHEPHERD-CONFIRMED] Native exchange SL successfully moved to ${suggested_sl:,.4f} on Bitget for {std_sym}")
                                        else:
                                            log.warning(f"[SHEPHERD-ABORT] modify_strategy_order returned false/empty for {std_sym}. Local SL not updated to prevent phantom stop.")
                                    except Exception as me:
                                        log.error(f"[SHEPHERD-ERROR] Failed to ratchet strategy SL on Bitget for {std_sym}: {me}")
                        else:
                            log.debug(f"[SHEPHERD] Suggested SL (${suggested_sl}) violates tightening bounds. Ignored.")

                    report_obj = {
                        "symbol": std_sym,
                        "side": pos_side,
                        "entry_price": entry_p,
                        "current_price": mark_p,
                        "unrealized_roe_pct": round(roe_pct, 2),
                        "r_multiple": r_multiple,
                        "holding_mins": round(holding_mins, 1),
                        "verdict": verdict,
                        "runner_mode": runner_mode,
                        "tape_condition": tape_cond,
                        "suggested_sl": suggested_sl,
                        "current_sl": current_sl,
                        "sl_ratcheted": sl_ratcheted,
                        "target_r": target_r,
                        "diagnosis": diagnosis,
                        "confidence": conf,
                        "evaluated_at": datetime.now(UTC).isoformat()
                    }
                    shepherd_reports.append(report_obj)
                    self._persist_shepherd_report(report_obj)

                return shepherd_reports
            finally:
                await client.exchange.close()

    def _update_local_sl(self, symbol: str, new_sl: float):
        """Updates local stop-loss in swing_open_positions and open_positions."""
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute(
                    "UPDATE swing_open_positions SET stop_loss = ?, trailing_stop = ?, updated_at = ? WHERE symbol = ?",
                    (new_sl, new_sl, datetime.now(UTC).isoformat(), symbol)
                )
                conn.execute(
                    "UPDATE open_positions SET stop_loss = ? WHERE symbol = ?",
                    (new_sl, symbol)
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Could not update local SL: {e}")

    def _persist_shepherd_report(self, rep: dict):
        """Writes report to system_state and trading_ledger.db."""
        clean_s = rep["symbol"].replace("/", "").replace("-", "").upper()
        now_iso = datetime.now(UTC).isoformat()

        # 1. Update system_state
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    (f"ai_shepherd_{clean_s}", json.dumps(rep), now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("latest_ai_shepherd_report", json.dumps(rep), now_iso)
                )
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to persist shepherd to system_state: {e}")

        # 2. Append to trading_ledger.db
        try:
            with sqlite3.connect("trading_ledger.db", timeout=5.0) as conn:
                conn.execute("""
                    INSERT INTO inflight_shepherd_logs (
                        timestamp, symbol, side, entry_price, current_price,
                        unrealized_roe_pct, r_multiple, holding_mins, verdict,
                        runner_mode, tape_condition, suggested_sl, current_sl,
                        sl_ratcheted, target_r, diagnosis, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now_iso, rep["symbol"], rep["side"], rep["entry_price"], rep["current_price"],
                    rep["unrealized_roe_pct"], rep["r_multiple"], rep["holding_mins"], rep["verdict"],
                    1 if rep["runner_mode"] else 0, rep["tape_condition"],
                    rep["suggested_sl"], rep["current_sl"], 1 if rep["sl_ratcheted"] else 0,
                    rep["target_r"], rep["diagnosis"], rep["confidence"]
                ))
                conn.commit()
        except Exception as e:
            log.warning(f"Failed to append to inflight_shepherd_logs: {e}")

    def _save_dormant_status(self):
        """Records dormant cash-shield state when 0 positions are active."""
        payload = {
            "status": "DORMANT_CASH_SHIELD",
            "active_count": 0,
            "message": "All capital held in 100% USDT Cash Shield. Shepherd dormant (zero token burn).",
            "updated_at": datetime.now(UTC).isoformat()
        }
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                    ("ai_shepherd_status", json.dumps(payload), datetime.now(UTC).isoformat())
                )
                conn.commit()
        except Exception:
            pass


# Singleton instance
_shepherd_instance = None

def get_shepherd() -> InflightPositionShepherd:
    global _shepherd_instance
    if _shepherd_instance is None:
        _shepherd_instance = InflightPositionShepherd()
    return _shepherd_instance

def run_single_shepherd_cycle(force: bool = False) -> list:
    """Convenience synchronous entry-point for background threads or supervisors."""
    shepherd = get_shepherd()
    return asyncio.run(shepherd.evaluate_active_positions(force=force))

if __name__ == "__main__":
    log.info("Starting Inflight Position Shepherd manual cycle...")
    res = run_single_shepherd_cycle(force=True)
    log.info(f"Shepherd cycle completed. Reports generated: {len(res)}")
    for r in res:
        print(json.dumps(r, indent=2))
