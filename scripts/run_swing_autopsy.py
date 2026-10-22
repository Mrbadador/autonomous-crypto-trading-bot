import asyncio
import json
import sqlite3
import logging
from ai_engine import query_cloud_llm, robust_json_decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SWING_AUTOPSY: %(message)s")
log = logging.getLogger("swing_autopsy")

async def run_swing_autopsy():
    conn_ledger = sqlite3.connect("trading_ledger.db", timeout=15)
    conn_data = sqlite3.connect("trading_data.db", timeout=15)

    # Clean out any unanalyzed placeholder entries from previous failed runs
    conn_ledger.execute("DELETE FROM trade_lessons WHERE bot_type = 'swing' AND lesson LIKE 'Forensic analysis completed for%'")
    conn_ledger.commit()

    try:
        cur_lessons = conn_ledger.execute("SELECT trade_id FROM trade_lessons WHERE bot_type = 'swing'")
        swing_analyzed_ids = {int(r[0]) for r in cur_lessons.fetchall() if r[0] is not None}
    except Exception:
        swing_analyzed_ids = set()

    cursor = conn_data.execute("""
        SELECT id, symbol, side, entry_price, exit_price, size, pnl, roe_pct, holding_hours, exit_reason, notes, timestamp
        FROM swing_trades
        ORDER BY id ASC
    """)
    swing_rows = cursor.fetchall()
    un_swing_rows = [r for r in swing_rows if int(r[0]) not in swing_analyzed_ids]

    log.info(f"Found {len(swing_rows)} total swing trades. Unanalyzed: {len(un_swing_rows)}.")

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

        log.info(f"--- Autopsying Trade #{t_id}: {symbol} {side.upper()} (PnL: ${pnl:+.2f}, ROE: {roe_pct:+.1f}%) ---")
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

        system_prompt = "You are an Elite Institutional Quant Forensic Officer."
        user_prompt = f"""Institutional Quant Autopsy on 4H Swing trade:
Trade: {symbol} {side.upper()}
Entry: {entry_p}, Exit: {exit_p}, PnL: {pnl:+.2f} USDT ({roe_pct:+.1f}% ROE).
Held: {holding_hours:.1f}h. Exit Reason: {exit_reason}. Notes: {notes}
Analyze root cause (e.g. unratcheted breakeven, correlation cluster, trend exhaustion, or successful runner trailing).
Output JSON only:
{{
  "lesson": "1-2 sentence forensic explanation",
  "learned_rule": "actionable quantitative rule",
  "rule_category": "SWING_DEFENSE" | "PORTFOLIO_CORRELATION" | "TRADE_MANAGEMENT" | "MOMENTUM" | "REGIME" | "RISK",
  "rule_confidence": 0.95
}}"""
        lesson_text = None
        for attempt in range(3):
            try:
                raw_resp = await query_cloud_llm(user_prompt, system_prompt)
                resp_json = robust_json_decode(raw_resp)
                if resp_json and isinstance(resp_json, dict) and "lesson" in resp_json:
                    lesson_text = resp_json.get("lesson")
                    log.info(f"   [LESSON] {lesson_text}")

                    if resp_json.get("learned_rule") and resp_json.get("rule_category"):
                        r_cat = str(resp_json["rule_category"]).upper()
                        r_txt = str(resp_json["learned_rule"]).strip()
                        r_conf = max(0.50, min(0.95, float(resp_json.get("rule_confidence") or 0.75)))

                        existing = conn_ledger.execute("SELECT 1 FROM learned_rules WHERE lower(rule_text) = lower(?)", (r_txt,)).fetchone()
                        if not existing:
                            # Deferred import: the learner module imports post_mortem_agent, so the
                            # shared writer is reached here rather than at module scope. This rule is
                            # an LLM's reading of one closed trade, so it is a lesson, not a measurement.
                            from backtests.batch_backtest_learner import write_learned_rule

                            write_learned_rule(
                                conn_ledger, r_cat, r_txt, r_conf,
                                source="swing_autopsy_llm_lesson",
                            )
                            conn_ledger.commit()
                            log.info(f"   [LEARNED RULE] [{r_cat}] {r_txt} (Conf: {r_conf:.0%})")
                        else:
                            log.info(f"   [EXISTING RULE] Rule already known in database.")
                    break
                else:
                    log.warning(f"   [WARN] Could not extract structured JSON lesson (attempt {attempt+1}): {raw_resp[:100]}")
                    await asyncio.sleep(2)
            except Exception as pe:
                log.error(f"Failed to extract forensic rule for swing trade {t_id} (attempt {attempt+1}): {pe}")
                await asyncio.sleep(2)

        if lesson_text:
            conn_ledger.execute(
                "INSERT INTO trade_lessons (trade_id, bot_type, symbol, pnl, original_rationale, lesson, timestamp) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (t_id, "swing", symbol, pnl, f"4H Swing {exit_reason} ({notes})", lesson_text)
            )
            conn_ledger.commit()
        else:
            log.warning(f"Skipping recording trade #{t_id} due to empty autopsy result.")

        await asyncio.sleep(4)

    log.info("=== Swing trades forensic autopsy sweep completed successfully! ===")
    conn_data.close()
    conn_ledger.close()

if __name__ == "__main__":
    asyncio.run(run_swing_autopsy())
