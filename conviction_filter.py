import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from ai_engine import red_team_deliberation

log = logging.getLogger("conviction_boardroom")

class ConvictionBoardroom:
    """
    V37.0 Multi-Agent Adversarial Conviction Boardroom.
    Deploys parallel Bearish Red Team Risk Officer vs. Bullish Alpha Hunter
    with a Supreme Arbiter synthesis (<600ms total) via Groq keypool.
    """
    async def get_conviction(self, symbol: str, side: str, context_data: dict) -> dict:
        # P1-1 FIX: Offload all initial synchronous SQLite reads off the event loop thread
        def _check_pre_filters_sync():
            res = {"veto": None}
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    # 1. Dynamic symbol whitelist in SQLite system_state
                    row = conn.execute("SELECT value FROM system_state WHERE key = 'ALLOWED_SYMBOLS'").fetchone()
                    if row and row[0]:
                        allowed = json.loads(row[0])
                        sym_clean = symbol.upper().replace("/", "").replace("-", "")
                        allowed_clean = [s.upper().replace("/", "").replace("-", "") for s in allowed]
                        if sym_clean not in allowed_clean:
                            res["veto"] = {
                                "execute": False,
                                "conviction_score": 0.0,
                                "final_rationale": f"Asset {symbol} filtered out — ALLOWED_SYMBOLS fleet constraint active."
                            }
                            res["log"] = f"[BOARDROOM] VETO: {symbol} is not in ALLOWED_SYMBOLS whitelist ({allowed}). Bypassing entry."
                            return res

                    # Check if this is a mean-reversion dip buy (oversold scalp)
                    is_dip_buy = (
                        context_data.get("strategy") == "Mean Reversion Maker Dip-Buy" or
                        (context_data.get("timeframe") == "5m" and float(context_data.get("rsi", 50.0) or 50.0) <= 30.0)
                    )

                    # 2. Bitcoin Anchor (Cross-Market Contagion Filter)
                    if "LONG" in side.upper() and symbol.upper() not in ["BTC/USDT", "BTCUSDT", "BTC"]:
                        row_pause = conn.execute("SELECT value FROM system_state WHERE key IN ('ALT_LONG_PAUSE', 'alt_long_pause')").fetchall()
                        row_state = conn.execute("SELECT value FROM system_state WHERE key = 'btc_anchor_state'").fetchone()
                        is_paused = any(r[0] in ("1", "true", "True") for r in row_pause)
                        btc_st = row_state[0] if row_state else "NEUTRAL_RANGE"
                        if is_paused or btc_st == "DISTRIBUTION_DUMP_RISK":
                            row_runners = conn.execute("SELECT value FROM system_state WHERE key = 'ai_decoupled_runners'").fetchone()
                            decoupled_runners = set(s.strip().upper().replace("/", "").replace("-", "") for s in (row_runners[0].split(",") if row_runners and row_runners[0] else []))
                            row_matrix = conn.execute("SELECT value FROM system_state WHERE key = 'ai_decoupled_regime_matrix'").fetchone()
                            if row_matrix and row_matrix[0]:
                                try:
                                    m_data = json.loads(row_matrix[0])
                                    for mk, mv in m_data.items():
                                        if isinstance(mv, dict) and mv.get("independent_trade_allowed"):
                                            decoupled_runners.add(mk.strip().upper().replace("/", "").replace("-", ""))
                                except Exception:
                                    pass
                            reg_row = conn.execute("SELECT plus_di, minus_di FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
                            buyers_dominate = bool(reg_row and float(reg_row[0] or 0.0) > float(reg_row[1] or 0.0))
                            sym_clean = symbol.upper().replace("/", "").replace("-", "")
                            
                            # Contagion protection: if Bitcoin is in severe systemic dump or contagion crisis,
                            # a minor 15m +DI micro-bounce must NOT bypass the altcoin pause.
                            is_severe_contagion = btc_st in ("DISTRIBUTION_DUMP_RISK", "CONTAGION_CRITICAL", "SYSTEMIC_DUMP", "EXTREME_CONTRACTION")
                            can_exempt = (sym_clean in decoupled_runners) or (buyers_dominate and not is_severe_contagion)
                            
                            if can_exempt:
                                res["decoupled_info"] = f"[BOARDROOM] 🚀 DECOUPLED ALPHA EXEMPTION: {symbol} authorized (Runner or Market +DI > -DI)."
                            elif is_dip_buy and not is_severe_contagion:
                                res["decoupled_info"] = f"[BOARDROOM] ⚡ MEAN-REVERSION DIP EXEMPTION: {symbol} is at severe oversold levels (RSI: {context_data.get('rsi')}). Dip-buy authorized despite Bitcoin Anchor pause."
                            else:
                                res["veto"] = {
                                    "execute": False,
                                    "conviction_score": 0.0,
                                    "final_rationale": f"Bitcoin Anchor Contagion Veto: BTC is under institutional distribution ({btc_st}). Alt longs paused."
                                }
                                res["log_warning"] = f"[BOARDROOM] ⚓ BITCOIN ANCHOR VETO: Altcoin long on {symbol} blocked. BTC state: {btc_st}, ALT_LONG_PAUSE=1."
                                return res

                    # 3. Multi-Asset Relative Strength Leader Gate (Avoid/Chop list)
                    # Only blocks breakout/momentum entries; mean-reversion dip-buys during chop are explicitly permitted
                    if not is_dip_buy:
                        row_avoid = conn.execute("SELECT value FROM system_state WHERE key = 'ai_asset_avoid'").fetchone()
                        if row_avoid and row_avoid[0]:
                            avoid_raw = row_avoid[0]
                            avoid_symbols = [s.strip().upper().replace("/", "").replace("-", "") for s in avoid_raw.split(",") if s.strip()]
                            sym_clean = symbol.upper().replace("/", "").replace("-", "")
                            if sym_clean in avoid_symbols:
                                res["veto"] = {
                                    "execute": False,
                                    "conviction_score": 0.0,
                                    "final_rationale": f"Leader Gate Veto: Asset {symbol} is currently flagged on the AI Avoid/Chop List. Capital reserved for alpha leaders."
                                }
                                res["log_warning"] = f"[BOARDROOM] 🚫 LEADER GATE VETO: {symbol} is in the AI Avoid/Chop List ({avoid_raw}). Vetoing entry."
                                return res
            except Exception as e:
                res["err"] = str(e)
                res["veto"] = {
                    "execute": False,
                    "conviction_score": 0.0,
                    "final_rationale": f"Boardroom Pre-Filter Veto (Fail-Closed on DB Error): {e}"
                }
                res["log_warning"] = f"[BOARDROOM] 🛡️ FAIL-CLOSED SAFETY VETO: Error reading system_state pre-filters: {e}"
            return res

        pre_filter_res = await asyncio.to_thread(_check_pre_filters_sync)
        if pre_filter_res.get("decoupled_info"):
            log.info(pre_filter_res["decoupled_info"])
        if pre_filter_res.get("log"):
            log.info(pre_filter_res["log"])
        if pre_filter_res.get("log_warning"):
            log.warning(pre_filter_res["log_warning"])
        if pre_filter_res.get("veto"):
            # Persist pre-filter veto into boardroom_decisions table for auditability
            def _log_prefilter_veto_sync():
                try:
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn_v:
                        conn_v.execute("""
                            CREATE TABLE IF NOT EXISTS boardroom_decisions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT NOT NULL,
                                symbol TEXT NOT NULL,
                                direction TEXT NOT NULL,
                                conviction_score REAL NOT NULL,
                                verdict TEXT NOT NULL,
                                agents_votes_json TEXT,
                                rationale TEXT,
                                subsequent_12h_pnl REAL,
                                counterfactual_outcome TEXT
                            )
                        """)
                        v_time = datetime.now(timezone.utc).isoformat()
                        v_rat = pre_filter_res["veto"].get("final_rationale", "Pre-Filter Veto")
                        conn_v.execute("""
                            INSERT INTO boardroom_decisions
                            (timestamp, symbol, direction, conviction_score, verdict, agents_votes_json, rationale, subsequent_12h_pnl, counterfactual_outcome)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            v_time,
                            str(symbol).upper(),
                            str(side).upper(),
                            0.0,
                            "VETOED",
                            json.dumps({"stage": "pre_filter", "error": pre_filter_res.get("err")}),
                            v_rat,
                            None,
                            "PENDING"
                        ))
                except Exception as ve:
                    log.debug(f"[BOARDROOM] Pre-filter veto persist note: {ve}")
            await asyncio.to_thread(_log_prefilter_veto_sync)
            return pre_filter_res["veto"]

        # Note: Learned-rules compliance and liquidity-sweep trap detection are consolidated
        # directly inside red_team_deliberation() to cut Boardroom LLM calls by 66% (3 calls -> 1 call).

        # [P0-RESTORED] Env-var-gated observation mode. Set OBSERVATION_MODE=1 in .env ONLY for
        # controlled data-harvesting sessions. Default is 0 (LIVE BOARDROOM ACTIVE).
        if os.getenv("OBSERVATION_MODE", "0") == "1":
            log.warning("[OBSERVATION MODE] Boardroom bypassed — OBSERVATION_MODE=1 is set. Set to 0 to restore live veto.")
            return {
                "execute": True,
                "conviction_score": 1.0,
                "final_rationale": "Observation Mode Override (OBSERVATION_MODE=1)"
            }

        symbol_upper = str(symbol).upper()
        log.info(f"[BOARDROOM] Convening Multi-Agent Red Team Deliberation for {symbol_upper} ({side.upper()})...")
        
        try:
            # Multi-Agent Adversarial Deliberation (Bearish Red Team vs Bullish Alpha Hunter + Arbiter)
            debate = await red_team_deliberation(symbol=symbol_upper, side=side, context_data=context_data, timeout_sec=12.0)
            
            raw_execute = bool(debate.get("execute", False))
            conv_score = float(debate.get("conviction_score", 0.0))
            red_obj = str(debate.get("red_team_objection", ""))
            alpha_th = str(debate.get("alpha_thesis", ""))
            raw_rationale = str(debate.get("final_rationale", "Adversarial deliberation rendered."))

            # Devil's Advocate Institutional Gate: Require >= 7.5 / 10 (0.75) conviction
            REQUIRED_CONVICTION = 0.75
            if raw_execute and conv_score >= REQUIRED_CONVICTION:
                execute = True
                rationale = raw_rationale
            else:
                execute = False
                if raw_execute and conv_score < REQUIRED_CONVICTION:
                    rationale = f"Devil's Advocate Veto: Conviction {conv_score*10:.1f}/10 below required 7.5/10 threshold. Bear objection: {red_obj}"
                    conv_score = 0.0
                else:
                    rationale = raw_rationale
                    conv_score = 0.0

            result = {
                "execute": execute,
                "conviction_score": conv_score,
                "final_rationale": rationale,
                "red_team_objection": red_obj,
                "alpha_thesis": alpha_th,
                "size_mult": debate.get("size_mult", 1.0),
                "sl_mult": debate.get("sl_mult", 2.0),
                "tp_mult": debate.get("tp_mult", 3.0)
            }
            
            # Persist debate to system_state and trading_ledger.db for live Dashboard auditability
            now_str = datetime.now(timezone.utc).isoformat()
            debate_record = {
                "symbol": symbol_upper,
                "side": side.upper(),
                "execute": execute,
                "conviction_score": conv_score,
                "red_team_objection": red_obj,
                "alpha_thesis": alpha_th,
                "final_rationale": rationale,
                "timestamp": now_str
            }

            def _persist_debate_sync():
                try:
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('latest_ai_debate', ?, ?)",
                            (json.dumps(debate_record), now_str)
                        )
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS boardroom_decisions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT NOT NULL,
                                symbol TEXT NOT NULL,
                                direction TEXT NOT NULL,
                                conviction_score REAL NOT NULL,
                                verdict TEXT NOT NULL,
                                agents_votes_json TEXT,
                                rationale TEXT,
                                subsequent_12h_pnl REAL,
                                counterfactual_outcome TEXT
                            )
                        """)
                        votes_json = json.dumps({
                            "red_team_objection": red_obj,
                            "alpha_thesis": alpha_th,
                            "raw_execute": raw_execute,
                            "size_mult": debate.get("size_mult", 1.0)
                        })
                        conn.execute("""
                            INSERT INTO boardroom_decisions 
                            (timestamp, symbol, direction, conviction_score, verdict, agents_votes_json, rationale, subsequent_12h_pnl, counterfactual_outcome)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            now_str,
                            symbol_upper,
                            side.upper(),
                            conv_score,
                            "APPROVED" if execute else "VETOED",
                            votes_json,
                            rationale,
                            None,
                            "PENDING"
                        ))
                except Exception as dbe:
                    log.debug(f"[BOARDROOM] State persist note: {dbe}")

                try:
                    with sqlite3.connect("trading_ledger.db", timeout=5.0) as conn_l:
                        conn_l.execute("""
                            CREATE TABLE IF NOT EXISTS boardroom_debates (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                symbol TEXT,
                                side TEXT,
                                execute INTEGER,
                                conviction_score REAL,
                                red_team_objection TEXT,
                                alpha_thesis TEXT,
                                final_rationale TEXT,
                                timestamp TEXT,
                                served_model_id TEXT
                            )
                        """)
                        # FIX 2: additive model provenance capture (read-only wrt decisioning).
                        if "served_model_id" not in [c[1] for c in conn_l.execute("PRAGMA table_info(boardroom_debates)").fetchall()]:
                            conn_l.execute("ALTER TABLE boardroom_debates ADD COLUMN served_model_id TEXT;")
                        served_model = debate.get("served_model_id")
                        conn_l.execute("""
                            INSERT INTO boardroom_debates (symbol, side, execute, conviction_score, red_team_objection, alpha_thesis, final_rationale, timestamp, served_model_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (symbol_upper, side.upper(), 1 if execute else 0, conv_score, red_obj, alpha_th, rationale, now_str, served_model))
                except Exception as l_err:
                    log.debug(f"[BOARDROOM] Ledger persist note: {l_err}")

            await asyncio.to_thread(_persist_debate_sync)

            log.info(f"[BOARDROOM] Arbiter Verdict for {symbol_upper}: Score={conv_score:.2f}, Execute={execute} | Red Team: {red_obj[:60]}... | Rationale: {rationale[:60]}...")
            return result
            
        except Exception as e:
            log.error(f"[BOARDROOM] Critical eval failure: {e}")
            # Fail CLOSED on any unhandled exception
            return {
                "execute": False,
                "conviction_score": 0.0,
                "final_rationale": "Boardroom internal error — Safety Veto. Check conviction_filter logs."
            }

