import sqlite3
import os
import time
import signal
import threading
import asyncio
import logging
import json
from ai_engine import query_cloud_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] POST_MORTEM: %(message)s")
log = logging.getLogger("post_mortem")

# P3-PATCH: Graceful shutdown + persistent event loop.
# Previously used asyncio.run() which creates and destroys a full event loop every
# hour — wasteful and incompatible with long-running coroutines. Now uses a single
# persistent loop for the lifetime of the process.
_stop_event = threading.Event()

def _handle_shutdown(signum, frame):
    log.info(f"[SHUTDOWN] Signal {signum} received. Post-Mortem Agent will exit after current sweep.")
    _stop_event.set()


def _write_ledger_rule(conn, category, rule_text, confidence, source):
    """Write one learned rule through the fleet's single shared writer.

    The import is function-local on purpose: `backtests/batch_backtest_learner.py` imports this
    module at module scope (`:25`), so a module-level import in the other direction is a cycle
    that breaks whichever module Python imports first.

    `source` is mandatory here rather than defaulted: an untagged row in `learned_rules` is
    indistinguishable from a hand-typed exemplar, and that is exactly the laundering the column
    exists to stop. The three producers below are all LLM readings of real closed ledger trades,
    which makes them lessons, not measurements — the text is model-authored.
    """
    from backtests.batch_backtest_learner import write_learned_rule

    write_learned_rule(conn, category, rule_text, confidence, source=source)


def init_db():
    conn = sqlite3.connect("trading_ledger.db")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                bot_type TEXT,
                symbol TEXT,
                pnl REAL,
                original_rationale TEXT,
                lesson TEXT,
                timestamp TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_autopsies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                bot_type TEXT,
                symbol TEXT,
                pnl REAL,
                timing_lesson TEXT,
                timing_rule TEXT,
                geometry_lesson TEXT,
                geometry_rule TEXT,
                counterfactual_lesson TEXT,
                counterfactual_rule TEXT,
                timestamp TEXT
            )
        """)
    conn.close()

# The only measured safe ranges this fleet has for its tunable knobs. Single source of truth
# for both the delta path below and the absolute-value path the learner writers use.
# Knobs deliberately absent (e.g. trail_dist) have no measured band and are refused.
PARAM_BOUNDS = {
    "rsi_buy": (10.0, 40.0),
    "rsi_sell": (60.0, 90.0),
    "take_profit_pct": (1.001, 1.05),
    "stop_loss_pct": (0.95, 0.999),
    "min_nn_confidence": (0.60, 0.95),
    "vol_gate_mult": (0.8, 3.0),
    "veto_threshold": (40.0, 95.0),
}


def enforce_parameter_bounds(param_name, current_val, suggested_adj):
    try:
        current_val = float(current_val)
        suggested_adj = float(suggested_adj)
    except (ValueError, TypeError):
        return current_val
    new_val = current_val + suggested_adj
    bounds = PARAM_BOUNDS.get(param_name)
    if bounds is None:
        # Unlisted knobs have no measured safe range: refuse rather than write an unbounded value.
        return current_val
    lo, hi = bounds
    return max(lo, min(hi, new_val))


def clamp_parameter_value(param_name, value):
    """Absolute value -> in-band value, or None when the knob has no measured safe range."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    bounds = PARAM_BOUNDS.get(param_name)
    if bounds is None:
        return None
    lo, hi = bounds
    return max(lo, min(hi, val))

async def analyze_trades():
    conn_ledger = None
    conn_data = None
    try:
        conn_ledger = sqlite3.connect("trading_ledger.db", timeout=15)
        conn_ledger.execute("PRAGMA journal_mode=WAL;")
        conn_ledger.execute("PRAGMA busy_timeout=15000;")
        
        conn_data = sqlite3.connect("trading_data.db", timeout=15)
        conn_data.execute("PRAGMA journal_mode=WAL;")
        conn_data.execute("PRAGMA busy_timeout=15000;")
        
        # 1. Analyze Sniper & Berserker (Legacy Logic)
        for table_prefix in ["", "berserker_"]:
            trades_table = f"{table_prefix}trades"
            bot_type = "berserker" if "berserker" in trades_table else "sniper"
            
            # Anti-join filtering in-memory
            try:
                cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = ?", (bot_type,))
                analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
            except Exception:
                analyzed_ids = set()

            try:
                cursor = conn_data.execute(f"""
                    SELECT id, symbol, side, entry_price, exit_price, pnl, regime, phase, ai_rationale
                    FROM {trades_table}
                    WHERE status = 'closed' OR status = 'CLOSED'
                """)
                rows = cursor.fetchall()
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    continue
                raise
            
            # Keep only unanalyzed rows
            un_rows = []
            for row in rows:
                try:
                    if int(row[0]) not in analyzed_ids:
                        un_rows.append(row)
                except (ValueError, TypeError):
                    pass
            
            for row in un_rows:
                try:
                    t_id = int(row[0])
                    symbol = row[1]
                    side = row[2]
                    entry_p = float(row[3]) if row[3] is not None else 0.0
                    exit_p = float(row[4]) if row[4] is not None else 0.0
                    pnl = float(row[5]) if row[5] is not None else 0.0
                    regime = row[6]
                    phase = row[7]
                    rationale = row[8]
                except (ValueError, TypeError) as e:
                    log.error(f"Float coercion failed for {bot_type} trade row {row}: {e}")
                    continue
                
                log.info(f"Analyzing {bot_type} trade {t_id} on {symbol} (PnL: {pnl:.2f})")
                
                # Fetch current params
                cur_params = conn_data.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = ?", (bot_type,))
                current_bot_params = dict(cur_params.fetchall())

                ctx = {
                    "bot_type": bot_type, "symbol": symbol, "side": side, "entry": entry_p, "exit": exit_p, "pnl": pnl,
                    "regime": regime, "phase": phase, "original_rationale": rationale,
                    "current_parameters": current_bot_params
                }

                system_prompt = "You are an Elite AI Quantitative Forensic Auditor. Your job is to extract an actionable post-mortem lesson from a completed crypto futures trade, conditionally suggest parameter tweaks, and distill concrete trading rules for our algorithmic memory."
                user_prompt = f"""Conduct a thorough post-mortem autopsy on this {bot_type} trade:
{json.dumps(ctx, indent=2)}

Parameters available to tweak: min_nn_confidence, vol_gate_mult, veto_threshold (berserker only).
Output exactly ONE JSON object (no markdown, no extra text) with this schema:
{{
  "lesson": "string, 1-2 sentence forensic explanation of what went wrong or went right",
  "suggest_tweak": boolean,
  "param_name": "string (one of the above) or null",
  "adjustment": float (how much to add/subtract, e.g. -0.05 for confidence, +0.1 for vol_gate_mult),
  "learned_rule": "string or null (a concise, actionable rule for future trade filtering, e.g. 'Avoid entering shorts when 1h RSI is under 25 and divergence is forming')",
  "rule_category": "MOMENTUM" | "REGIME" | "VOLATILITY" | "RISK" | null,
  "rule_confidence": float between 0.60 and 0.95
}}"""
                lesson_text = "Analyzed trade."
                try:
                    raw_resp = await query_cloud_llm(user_prompt, system_prompt)
                    from ai_engine import robust_json_decode
                    resp_json = robust_json_decode(raw_resp)
                    if not resp_json or not isinstance(resp_json, dict):
                        raise ValueError("LLM did not return a valid JSON object for lesson extraction")
                    lesson_text = resp_json.get("lesson", lesson_text)

                    # Apply Tweak
                    if resp_json.get("suggest_tweak") and resp_json.get("param_name") in current_bot_params:
                        p_name = resp_json["param_name"]
                        p_adj = resp_json.get("adjustment", 0)
                        cur_val = current_bot_params[p_name]
                        new_val = enforce_parameter_bounds(p_name, cur_val, p_adj)
                        
                        if new_val != cur_val:
                            conn_data.execute("UPDATE bot_parameters SET param_value = ?, last_updated = datetime('now') WHERE bot_name = ? AND param_name = ?", (new_val, bot_type, p_name))
                            conn_data.commit()
                            log.info(f"AI Self-Optimization ({bot_type}): Tweaked {p_name} from {cur_val} to {new_val}")
                            lesson_text += f" [Optimized {p_name} by {p_adj}]"

                    # Distill High-Confidence Learned Rule into Long-Term Memory
                    if resp_json.get("learned_rule") and resp_json.get("rule_category"):
                        r_cat = str(resp_json["rule_category"]).upper()
                        r_txt = str(resp_json["learned_rule"]).strip()
                        r_conf = max(0.50, min(0.95, float(resp_json.get("rule_confidence") or 0.75)))
                        try:
                            # Verify if an identical or highly similar rule already exists in memory
                            exists = conn_ledger.execute(
                                "SELECT id FROM learned_rules WHERE category = ? AND rule_text = ?",
                                (r_cat, r_txt)
                            ).fetchone()
                            if not exists:
                                _write_ledger_rule(
                                    conn_ledger, r_cat, r_txt, r_conf,
                                    source="post_mortem_llm_lesson",
                                )
                                conn_ledger.commit()
                                log.info(f"[POST-MORTEM-RULE] 🧠 Added Dynamic Learned Rule: [{r_cat}] {r_txt} (Conf: {r_conf:.0%})")
                            else:
                                log.debug(f"[POST-MORTEM-RULE] Duplicate rule already present in ledger: [{r_cat}] {r_txt}")
                        except Exception as re_err:
                            log.warning(f"Error inserting learned rule: {re_err}")

                except Exception as e:
                    log.error(f"Failed to optimize {bot_type} trade {t_id}: {e}")
                
                conn_ledger.execute("INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))", (t_id, bot_type, symbol, pnl, rationale, lesson_text))
                conn_ledger.commit()
                await asyncio.sleep(1)

        # 2. Analyze Mean-Reversion for Self-Optimization Loop
        try:
            cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = 'mean_reversion'")
            mr_analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
        except Exception:
            mr_analyzed_ids = set()

        try:
            cursor = conn_data.execute("SELECT rowid, symbol, side, entry_price, exit_price, pnl FROM mean_reversion_trades WHERE status = 'closed' OR status = 'CLOSED'")
            mr_rows = cursor.fetchall()
        except sqlite3.OperationalError:
            mr_rows = []

        un_mr_rows = []
        for row in mr_rows:
            try:
                if int(row[0]) not in mr_analyzed_ids:
                    un_mr_rows.append(row)
            except (ValueError, TypeError):
                pass

        for row in un_mr_rows:
            try:
                t_id = int(row[0])
                symbol = row[1]
                side = row[2]
                entry_p = float(row[3]) if row[3] is not None else 0.0
                exit_p = float(row[4]) if row[4] is not None else 0.0
                pnl = float(row[5]) if row[5] is not None else 0.0
            except (ValueError, TypeError) as e:
                log.error(f"Float coercion failed for mean_reversion trade row {row}: {e}")
                continue
                
            log.info(f"Analyzing mean_reversion trade {t_id} on {symbol} (PnL: {pnl:.2f})")
            
            # Fetch current params
            cur_params = conn_data.execute("SELECT param_name, param_value FROM bot_parameters WHERE bot_name = 'mean_reversion'")
            current_bot_params = dict(cur_params.fetchall())
            
            ctx = {
                "bot_type": "mean_reversion", "symbol": symbol, "side": side, "entry": entry_p, "exit": exit_p, "pnl": pnl,
                "current_parameters": current_bot_params
            }
            
            system_prompt = "You are an AI Trading Optimizer. Your job is to extract a lesson from a trade and conditionally suggest a parameter tweak."
            user_prompt = f"""Review this mean_reversion trade:
{json.dumps(ctx, indent=2)}

Parameters available to tweak: rsi_buy, rsi_sell, take_profit_pct, stop_loss_pct.
Output exactly ONE JSON object (no markdown, no extra text) with this schema:
{{
  "lesson": "string, 1-sentence explanation of what went wrong or went right",
  "suggest_tweak": boolean,
  "param_name": "string (one of the above) or null",
  "adjustment": float (how much to add/subtract, e.g. -0.005 for stop loss, +2 for rsi_buy, or 0)
}}"""
            lesson = "Analyzed trade."
            try:
                raw_resp = await query_cloud_llm(user_prompt, system_prompt)
                resp_json = json.loads(raw_resp.replace('```json', '').replace('```', '').strip())
                lesson = resp_json.get("lesson", lesson)
                
                # Apply Tweak
                if resp_json.get("suggest_tweak") and resp_json.get("param_name") in current_bot_params:
                    p_name = resp_json["param_name"]
                    p_adj = resp_json.get("adjustment", 0)
                    cur_val = current_bot_params[p_name]
                    new_val = enforce_parameter_bounds(p_name, cur_val, p_adj)
                    
                    if new_val != cur_val:
                        conn_data.execute("UPDATE bot_parameters SET param_value = ?, last_updated = datetime('now') WHERE bot_name = 'mean_reversion' AND param_name = ?", (new_val, p_name))
                        conn_data.commit()
                        log.info(f"AI Self-Optimization: Tweaked {p_name} from {cur_val} to {new_val}")
                        lesson += f" [Optimized {p_name} by {p_adj}]"
            except Exception as e:
                log.error(f"Failed to parse or apply optimization: {e}")
                lesson = "Failed to parse AI optimization response."

            conn_ledger.execute("INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))", (t_id, "mean_reversion", symbol, pnl, "Mean-Reversion Strategy", lesson))
            conn_ledger.commit()
            await asyncio.sleep(2)

        # 3. Analyze MARL Trades
        try:
            cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = 'marl'")
            marl_analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
        except Exception:
            marl_analyzed_ids = set()

        try:
            cursor = conn_data.execute("""
                SELECT id, symbol, side, entry_price, exit_price, pnl, regime, phase, ai_rationale 
                FROM marl_live_trades 
                WHERE status = 'closed' OR status = 'CLOSED'
            """)
            marl_rows = cursor.fetchall()
        except sqlite3.OperationalError:
            marl_rows = []

        un_marl_rows = []
        for row in marl_rows:
            try:
                if int(row[0]) not in marl_analyzed_ids:
                    un_marl_rows.append(row)
            except (ValueError, TypeError):
                pass

        for row in un_marl_rows:
            try:
                t_id = int(row[0])
                symbol = row[1]
                side = row[2]
                entry_p = float(row[3]) if row[3] is not None else 0.0
                exit_p = float(row[4]) if row[4] is not None else 0.0
                pnl = float(row[5]) if row[5] is not None else 0.0
                regime = row[6]
                phase = row[7]
                rationale = row[8]
            except (ValueError, TypeError) as e:
                log.error(f"Float coercion failed for MARL trade row {row}: {e}")
                continue

            log.info(f"Analyzing marl trade {t_id} on {symbol} (PnL: {pnl:.2f})")
            
            ctx = {
                "bot_type": "marl", "symbol": symbol, "side": side, "entry": entry_p, "exit": exit_p, "pnl": pnl,
                "regime": regime, "phase": phase, "original_rationale": rationale
            }
            
            system_prompt = "You are an AI Trading Optimizer. Your job is to extract a lesson from a trade."
            user_prompt = f"""Review this marl trade:
{json.dumps(ctx, indent=2)}

Output exactly ONE JSON object (no markdown, no extra text) with this schema:
{{
  "lesson": "string, 1-sentence explanation of what went wrong or went right"
}}"""
            lesson = "Analyzed trade."
            try:
                raw_resp = await query_cloud_llm(user_prompt, system_prompt)
                resp_json = json.loads(raw_resp.replace('```json', '').replace('```', '').strip())
                lesson = resp_json.get("lesson", lesson)
            except Exception as e:
                log.error(f"Failed to parse MARL post-mortem response: {e}")
                lesson = "Failed to parse AI post-mortem response."

            conn_ledger.execute("INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))", (t_id, "marl", symbol, pnl, rationale, lesson))
            conn_ledger.commit()
            await asyncio.sleep(2)

        # 4. Analyze Live Paper Trades from live_paper_trades.db
        if os.path.exists("live_paper_trades.db"):
            conn_paper = None
            try:
                conn_paper = sqlite3.connect("live_paper_trades.db", timeout=15)
                conn_paper.execute("PRAGMA journal_mode=WAL;")
                conn_paper.execute("PRAGMA busy_timeout=15000;")
                
                for table_name, bot_type in [("trades", "paper_sniper"), ("berserker_trades", "paper_berserker")]:
                    try:
                        cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = ?", (bot_type,))
                        analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
                    except Exception:
                        analyzed_ids = set()

                    try:
                        cursor = conn_paper.execute(f"""
                            SELECT id, symbol, side, entry_price, exit_price, pnl, regime, phase, notes
                            FROM {table_name}
                            WHERE action IN ('sl_hit', 'tp_hit', 'partial_tp', 'closed', 'close')
                            ORDER BY id DESC LIMIT 15
                        """)
                        paper_rows = cursor.fetchall()
                    except sqlite3.OperationalError:
                        paper_rows = []

                    un_paper_rows = [r for r in paper_rows if int(r[0]) not in analyzed_ids]
                    for r in un_paper_rows:
                        t_id = int(r[0])
                        symbol = r[1]
                        side = r[2]
                        entry_p = float(r[3] or 0.0)
                        exit_p = float(r[4] or 0.0)
                        pnl = float(r[5] or 0.0)
                        regime = r[6] or "UNKNOWN"
                        phase = r[7] or "UNKNOWN"
                        rationale = r[8] or ""

                        log.info(f"Analyzing {bot_type} trade #{t_id} on {symbol} (PnL: ${pnl:+.2f})")
                        ctx = {
                            "bot_type": bot_type, "symbol": symbol, "side": side,
                            "entry": entry_p, "exit": exit_p, "pnl": pnl,
                            "regime": regime, "phase": phase, "notes": rationale
                        }

                        system_prompt = "You are an Elite AI Quantitative Forensic Auditor. Extract an actionable post-mortem lesson from a completed crypto futures trade, and distill a concrete trading rule for our algorithmic memory."
                        user_prompt = f"""Conduct a forensic post-mortem autopsy on this {bot_type} trade:
{json.dumps(ctx, indent=2)}

Output exactly ONE JSON object (no markdown, no extra text) with this schema:
{{
  "lesson": "string, 1-2 sentence forensic explanation of what went wrong or went right",
  "learned_rule": "string or null (a concise, actionable rule for future trade filtering)",
  "rule_category": "MOMENTUM" | "REGIME" | "VOLATILITY" | "RISK" | null,
  "rule_confidence": float between 0.60 and 0.95
}}"""
                        lesson_text = f"Forensic analysis completed for {symbol} ({pnl:+.2f} USDT)."
                        try:
                            raw_resp = await query_cloud_llm(user_prompt, system_prompt)
                            from ai_engine import robust_json_decode
                            resp_json = robust_json_decode(raw_resp)
                            if resp_json and isinstance(resp_json, dict):
                                lesson_text = resp_json.get("lesson", lesson_text)
                                if resp_json.get("learned_rule") and resp_json.get("rule_category"):
                                    r_cat = str(resp_json["rule_category"]).upper()
                                    r_txt = str(resp_json["learned_rule"]).strip()
                                    r_conf = max(0.50, min(0.95, float(resp_json.get("rule_confidence") or 0.75)))

                                    existing = conn_ledger.execute("SELECT 1 FROM learned_rules WHERE lower(rule_text) = lower(?)", (r_txt,)).fetchone()
                                    if not existing:
                                        _write_ledger_rule(
                                            conn_ledger, r_cat, r_txt, r_conf,
                                            source="post_mortem_llm_lesson",
                                        )
                                        conn_ledger.commit()
                                        log.info(f"[POST-MORTEM-RULE] 🧠 Added Dynamic Learned Rule: [{r_cat}] {r_txt} (Conf: {r_conf:.0%})")
                        except Exception as pe:
                            log.error(f"Failed to extract forensic rule for {bot_type} trade {t_id}: {pe}")

                        conn_ledger.execute(
                            "INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                            (t_id, bot_type, symbol, pnl, rationale, lesson_text)
                        )
                        conn_ledger.commit()
                        await asyncio.sleep(1)
            except Exception as pe:
                log.error(f"Error in paper trades post-mortem: {pe}")
            finally:
                if conn_paper:
                    try: conn_paper.close()
                    except Exception: pass

        # 5. Analyze Live 4H Swing Architect Trades from swing_trades
        try:
            cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = 'swing'")
            swing_analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
        except Exception:
            swing_analyzed_ids = set()

        try:
            cursor = conn_data.execute("""
                SELECT id, symbol, side, entry_price, exit_price, size, pnl, roe_pct, holding_hours, exit_reason, notes, timestamp
                FROM swing_trades
                ORDER BY id ASC
            """)
            swing_rows = cursor.fetchall()
        except sqlite3.OperationalError as se:
            log.warning(f"Could not query swing_trades: {se}")
            swing_rows = []

        un_swing_rows = [r for r in swing_rows if int(r[0]) not in swing_analyzed_ids]
        for r in un_swing_rows:
            t_id = int(r[0])
            symbol = r[1]
            side = r[2]
            entry_p = float(r[3] or 0.0)
            exit_p = float(r[4] or 0.0)
            size = float(r[5] or 0.0)
            pnl = float(r[6] or 0.0)
            roe_pct = float(r[7] or 0.0)
            holding_hours = float(r[8] or 0.0)
            exit_reason = r[9] or "UNKNOWN"
            notes = r[10] or ""
            t_stamp = r[11] or ""

            log.info(f"Analyzing 4H Swing Architect trade #{t_id} on {symbol} (PnL: ${pnl:+.2f}, ROE: {roe_pct:+.1f}%)")
            ctx = {
                "engine": "The Swing Architect (4H Trend Engine)",
                "trade_id": t_id,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_p,
                "exit_price": exit_p,
                "notional_size": round(size * entry_p, 2),
                "pnl_usdt": round(pnl, 4),
                "roe_pct": round(roe_pct, 2),
                "holding_hours": round(holding_hours, 1),
                "exit_reason": exit_reason,
                "notes": notes,
                "timestamp": t_stamp
            }

            # V38.0 Counterfactual 3-Agent Institutional Autopsy Suite
            timing_prompt = (
                f"You are the Timing & Breakout Confirmation Forensic Agent.\n"
                f"Evaluate this {symbol} {side.upper()} trade (Entry: {entry_p}, Exit: {exit_p}, PnL: ${pnl:+.2f}, Held: {holding_hours:.1f}h, Exit: {exit_reason}, Notes: {notes}).\n"
                "Evaluate whether entry was premature, chasing an extended candle, or cleanly confirmed by HTF structure.\n"
                "Return strictly JSON: {\"lesson\": \"1 sentence on timing\", \"learned_rule\": \"concise actionable rule for entry timing\", \"rule_confidence\": 0.85}"
            )

            geometry_prompt = (
                f"You are the Trade Geometry & Volatility Stop Forensic Agent.\n"
                f"Evaluate this {symbol} {side.upper()} trade (Entry: {entry_p}, Exit: {exit_p}, PnL: ${pnl:+.2f}, Held: {holding_hours:.1f}h, Exit: {exit_reason}, Notes: {notes}).\n"
                "Evaluate ATR stop-loss distance, volatility buffer, and whether the trade had adequate breathing room (Rule #173).\n"
                "Return strictly JSON: {\"lesson\": \"1 sentence on stop geometry\", \"learned_rule\": \"concise actionable rule for stop calibration\", \"rule_confidence\": 0.85}"
            )

            counterfactual_prompt = (
                f"You are the Counterfactual 'What If?' Strategy Simulation Agent.\n"
                f"Evaluate this {symbol} {side.upper()} trade (Entry: {entry_p}, Exit: {exit_p}, PnL: ${pnl:+.2f}, Held: {holding_hours:.1f}h, Exit: {exit_reason}, Notes: {notes}).\n"
                "Simulate what would have happened if breakeven was ratcheted earlier/later, or if trailing stop used 4H Kijun vs 15m EMA.\n"
                "Return strictly JSON: {\"lesson\": \"1 sentence on counterfactual simulation\", \"learned_rule\": \"concise actionable rule for trailing management\", \"rule_confidence\": 0.85}"
            )

            sys_p = "You are an Elite Institutional AI Quantitative Forensic Officer. Return strictly valid JSON."

            async def _run_agent(prompt_text):
                try:
                    from ai_engine import query_cloud_llm, robust_json_decode
                    raw = await query_cloud_llm(prompt_text, sys_p)
                    return robust_json_decode(raw) or {}
                except Exception as e:
                    log.warning(f"Forensic sub-agent query error: {e}")
                    return {}

            autopsy_results = await asyncio.gather(
                _run_agent(timing_prompt),
                _run_agent(geometry_prompt),
                _run_agent(counterfactual_prompt),
                return_exceptions=True
            )

            t_res = autopsy_results[0] if isinstance(autopsy_results[0], dict) else {}
            g_res = autopsy_results[1] if isinstance(autopsy_results[1], dict) else {}
            c_res = autopsy_results[2] if isinstance(autopsy_results[2], dict) else {}

            # Distill up to 3 high-confidence rules into learned_rules table
            rule_triplets = [
                ("TIMING_CONFIRMATION", t_res.get("learned_rule"), t_res.get("rule_confidence", 0.80)),
                ("GEOMETRY_STOP", g_res.get("learned_rule"), g_res.get("rule_confidence", 0.85)),
                ("COUNTERFACTUAL_SIM", c_res.get("learned_rule"), c_res.get("rule_confidence", 0.80)),
            ]

            for cat, r_text, r_conf in rule_triplets:
                if r_text and len(str(r_text).strip()) > 10:
                    r_clean = str(r_text).strip()
                    conf_val = max(0.60, min(0.95, float(r_conf or 0.80)))
                    try:
                        existing = conn_ledger.execute("SELECT 1 FROM learned_rules WHERE lower(rule_text) = lower(?)", (r_clean,)).fetchone()
                        if not existing:
                            _write_ledger_rule(
                                conn_ledger, cat, r_clean, conf_val,
                                source="three_agent_autopsy_lesson",
                            )
                            conn_ledger.commit()
                            log.info(f"[3-AGENT-AUTOPSY] 🧠 Added Dynamic Rule [{cat}]: {r_clean} (Conf: {conf_val:.0%})")
                    except Exception as ins_err:
                        log.debug(f"Rule insert note: {ins_err}")

            # Persist 3-Agent Autopsy to trade_autopsies table
            try:
                conn_ledger.execute("""
                    CREATE TABLE IF NOT EXISTS trade_autopsies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        bot_type TEXT,
                        symbol TEXT,
                        pnl REAL,
                        timing_lesson TEXT,
                        timing_rule TEXT,
                        geometry_lesson TEXT,
                        geometry_rule TEXT,
                        counterfactual_lesson TEXT,
                        counterfactual_rule TEXT,
                        timestamp TEXT
                    )
                """)
                conn_ledger.execute("""
                    INSERT INTO trade_autopsies
                    (trade_id, bot_type, symbol, pnl, timing_lesson, timing_rule, geometry_lesson, geometry_rule, counterfactual_lesson, counterfactual_rule, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (
                    t_id, "swing", symbol, pnl,
                    t_res.get("lesson", ""), t_res.get("learned_rule", ""),
                    g_res.get("lesson", ""), g_res.get("learned_rule", ""),
                    c_res.get("lesson", ""), c_res.get("learned_rule", ""),
                ))
                conn_ledger.commit()
            except Exception as auto_err:
                log.debug(f"Trade autopsies table persist note: {auto_err}")

            unified_lesson = f"Timing: {t_res.get('lesson', 'N/A')} | Geometry: {g_res.get('lesson', 'N/A')} | Counterfactual: {c_res.get('lesson', 'N/A')}"
            conn_ledger.execute(
                "INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (t_id, "swing", symbol, pnl, f"4H Swing {exit_reason} ({notes})", unified_lesson)
            )
            conn_ledger.commit()
            log.info(f"[3-AGENT-AUTOPSY] Completed comprehensive autopsy for trade #{t_id} ({symbol}).")
            await asyncio.sleep(1)

        # 6. Analyze Ghost / Shadow Trades (MED-06: Filter Alpha Attribution)
        try:
            cur_ghosts = conn_data.execute("SELECT outcome, pnl FROM ghost_trades WHERE outcome IS NOT NULL")
            g_rows = cur_ghosts.fetchall()
            if g_rows:
                total_ghosts = len(g_rows)
                losses_avoided = sum(1 for r in g_rows if r[0] == 'sl_hit' or (r[1] is not None and float(r[1]) < 0))
                accuracy = (losses_avoided / total_ghosts) * 100.0
                conn_data.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('filter_alpha_score', ?, datetime('now'))",
                    (f"{accuracy:.1f}% ({losses_avoided}/{total_ghosts} losses avoided)",)
                )
                conn_data.commit()
                log.info(f"[GHOST-AUDIT] Filter Alpha Score: {accuracy:.1f}% ({losses_avoided}/{total_ghosts} bad trades blocked).")
        except Exception as ge:
            log.debug(f"[GHOST-AUDIT] Skip note: {ge}")

    except Exception as e:
        log.error(f"Error in post-mortem analysis: {e}")
    finally:
        if conn_data is not None:
            try:
                conn_data.close()
            except Exception:
                pass
        if conn_ledger is not None:
            try:
                conn_ledger.close()
            except Exception:
                pass


async def trigger_instant_loss_autopsy(symbol: str, pnl: float, exit_reason: str = "SL_HIT", trade_id: int = None) -> dict:
    """
    Feature 5: Instant AI Loss Autopsy & Dynamic Rule Generator.
    Conducts immediate forensic autopsy upon trade stop-out and distills fresh learned rules.
    """
    log.info(f"[INSTANT-AUTOPSY] 🚨 Triggered instant loss autopsy for {symbol} (PnL: ${pnl:+.4f}, Reason: {exit_reason})")
    try:
        await analyze_trades()
        return {"status": "completed", "symbol": symbol, "pnl": pnl}
    except Exception as e:
        log.error(f"[INSTANT-AUTOPSY-ERR] Error in instant autopsy: {e}")
        return {"status": "error", "error": str(e)}


def run_agent():
    init_db()
    # Registered here, never at import time: backtests/batch_backtest_learner.py and
    # scripts/deep_brain_trainer.py import this module at module scope and swing_trader.py
    # imports it lazily inside its position-close path, so an import-time registration used to
    # overwrite the importer's own SIGINT/SIGTERM handler and disarm graceful shutdown.
    # run_agent() only ever runs on the main thread (this module's __main__, and the
    # `python post_mortem_agent.py` subprocesses spawned by main.py and start_system.py).
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    # P3-PATCH: Persistent event loop — avoids creating/destroying a loop every hour.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not _stop_event.is_set():
            log.info("Starting post-mortem forensic sweep...")
            loop.run_until_complete(analyze_trades())
            log.info("Post-mortem sweep complete. Sleeping for 60 seconds (reactive watch).")
            # 60s reactive sweep: checks for newly closed trades immediately
            _stop_event.wait(timeout=60)
    finally:
        loop.close()
        log.info("[SHUTDOWN] Post-Mortem Agent exiting cleanly.")

if __name__ == "__main__":
    run_agent()
