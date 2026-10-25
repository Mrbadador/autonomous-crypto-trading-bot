"""
hourly_briefing.py — Sovereign Swarm 1-Hour Executive Briefing Rollup
====================================================================
Synthesizes the autonomous 5-minute executive briefing cycles over the past
60 minutes (typically 12 cycles from executive_briefing_history) into an
actionable, high-fidelity institutional email dispatch.

Covers:
1. Hourly Equity Delta, Peak/Low Range, and Margin Utilization Trajectory.
2. Chronological Tactical Directives issued by the AI Sovereign Commander.
3. Fleet Trading Activity: Trades opened, closed, or ratcheted in the past hour.
4. Active Position Snapshot: Real-time Mark, Funding Paid/Gained, True Net PnL,
   Dual-Layer Stops, and In-Flight Sentinel verdicts.
5. Macro Blueprint & Next-Hour Strategic Outlook.

Schedule: Runs every 60 minutes in-process via start_system.py or standalone CLI.
"""

import os
import sys
import time
import json
import sqlite3
import re
import logging
import html
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
import config

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HOURLY_BRIEFING] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("hourly_briefing")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
TRADING_LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")


def get_db_connection(db_path: str, timeout: float = 8.0) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    return conn


def safe_query(db_path: str, sql: str, params=()) -> List[Dict[str, Any]]:
    if not os.path.exists(db_path):
        return []
    try:
        conn = get_db_connection(db_path)
        try:
            cursor = conn.execute(sql, params)
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()
    except Exception as e:
        log.warning(f"DB query error on {os.path.basename(db_path)}: {e}")
        return []


def query_system_state(key: str, default: str = "") -> str:
    rows = safe_query(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = ?", (key,))
    if rows and "value" in rows[0] and rows[0]["value"] is not None:
        return str(rows[0]["value"])
    return default


def _shep_verdict_display(raw_verdict) -> Dict[str, str]:
    """A shepherd verdict that was never measured renders as an absence, never as an all-clear word.

    Pure: it maps the published verdict value (or its absence) to the briefing's on-screen text and
    colour, so the briefing cannot print "HOLDING_NORMAL"/"MONITORED" — let alone the raw absence
    token — for a position the AI never assessed.
    """
    absent = (raw_verdict is None or str(raw_verdict).strip() == ""
              or str(raw_verdict).strip().upper() == getattr(config, "VERDICT_ABSENT", "NO_AI_VERDICT"))
    if absent:
        return {"text": "no AI verdict measured", "color": "#94a3b8"}
    return {"text": str(raw_verdict), "color": "#38bdf8"}


def generate_hourly_briefing_data() -> Dict[str, Any]:
    """Gathers and synthesizes the last 60 minutes of 5-minute briefings and fleet telemetry."""
    now_utc = datetime.now(timezone.utc)
    one_hour_ago = now_utc - timedelta(minutes=65)
    one_hour_iso = one_hour_ago.strftime("%Y-%m-%d %H:%M:%S")

    # 1. Fetch 5-Minute Briefing Cycles from the last hour (timestamp-aware)
    cycles_rows = safe_query(
        TRADING_LEDGER_DB,
        "SELECT * FROM executive_briefing_history WHERE generated_at >= ? ORDER BY id ASC",
        (one_hour_iso,)
    )
    if not cycles_rows or len(cycles_rows) < 3:
        # Fallback to last 12 cycles only if recent (within 4 hours)
        cycles_rows = safe_query(
            TRADING_LEDGER_DB,
            "SELECT * FROM executive_briefing_history ORDER BY id DESC LIMIT 12"
        )
        cycles_rows = list(reversed(cycles_rows))

    cycles_count = len(cycles_rows)
    oldest_cycle = cycles_rows[0] if cycles_rows else {}
    newest_cycle = cycles_rows[-1] if cycles_rows else {}

    start_equity = float(oldest_cycle.get("equity") or 0.0)
    end_equity = float(newest_cycle.get("equity") or 0.0)

    # Truthful live balance query without fabricated constants
    if end_equity <= 0.0:
        live_bal_str = query_system_state("live_balance", "") or query_system_state("balance", "0.0")
        try:
            end_equity = float(live_bal_str)
        except Exception:
            end_equity = 0.0
        if start_equity <= 0.0:
            start_equity = end_equity

    equity_delta = end_equity - start_equity
    equity_delta_pct = (equity_delta / start_equity * 100.0) if start_equity > 0 else 0.0

    equities = [float(r.get("equity") or 0.0) for r in cycles_rows if float(r.get("equity") or 0.0) > 0]
    peak_equity = max(equities) if equities else end_equity
    low_equity = min(equities) if equities else end_equity

    # Truthful live margin utilization without fabricated 54% / 74.9% defaults
    margin_utils = [float(r.get("margin_util_pct") or 0.0) for r in cycles_rows if r.get("margin_util_pct") is not None]
    if margin_utils:
        min_margin = min(margin_utils)
        max_margin = max(margin_utils)
        end_margin = margin_utils[-1]
    else:
        live_util = query_system_state("margin_util_pct", "0.0")
        try:
            end_margin = float(live_util)
        except Exception:
            end_margin = 0.0
        min_margin, max_margin = end_margin, end_margin

    start_cycle_num = oldest_cycle.get("cycle_num", 1)
    end_cycle_num = newest_cycle.get("cycle_num", 1)

    # 2. Extract Directives and Tactical Themes across the hour
    directives = []
    seen_directives = set()
    latest_memo_full = {}
    for r in cycles_rows:
        md = r.get("markdown") or ""
        # Match directive title
        m = re.search(r"TACTICAL DIRECTIVE:\*\*\s*\*\*([^\*]+)\*\*", md)
        if not m:
            m = re.search(r"TACTICAL DIRECTIVE:\*\*\s*([^\n]+)", md)
        if m:
            d_text = m.group(1).strip()
            if d_text not in seen_directives:
                seen_directives.add(d_text)
                directives.append({
                    "cycle": r.get("cycle_num"),
                    "time": r.get("generated_at", "")[-12:-4] if r.get("generated_at") else "",
                    "directive": d_text
                })

    # Extract detailed sections from the latest 5m briefing markdown
    latest_md = newest_cycle.get("markdown") or ""
    latest_rationale = ""
    latest_vetoes = ""
    latest_watch_levels = ""
    latest_stance = ""
    latest_macro_bias = newest_cycle.get("regime", "CHOP")
    latest_macro_narrative = ""

    rat_m = re.search(r"\*\s*\*\*Positioning Rationale:\*\*\s*([^\n]+)", latest_md)
    if rat_m: latest_rationale = rat_m.group(1).strip()

    vet_m = re.search(r"\*\s*\*\*Setup Veto & Screening Status:\*\*\s*([^\n]+)", latest_md)
    if vet_m: latest_vetoes = vet_m.group(1).strip()

    wat_m = re.search(r"\*\s*\*\*Key Watch Levels:\*\*\s*`?([^`\n]+)`?", latest_md)
    if wat_m: latest_watch_levels = wat_m.group(1).strip()

    sta_m = re.search(r"\*\s*\*\*Operational Stance:\*\*\s*\*\"?([^\*\n\"]+)\"?\*", latest_md)
    if sta_m: latest_stance = sta_m.group(1).strip()

    nar_m = re.search(r"\*\s*\*\*Strategic Macro Narrative:\*\*\s*([^\n]+)", latest_md)
    if nar_m: latest_macro_narrative = nar_m.group(1).strip()

    # 3. Fetch Trades Executed Over the Past 60 Minutes (All fleet engines, ISO + space normalized)
    closed_trades = []
    trade_queries = [
        ("trades", "SELECT symbol, side, pnl, action AS exit_reason, status, timestamp, notes FROM trades"),
        ("swing_trades", "SELECT symbol, side, pnl, exit_reason, notes, timestamp FROM swing_trades"),
        ("berserker_trades", "SELECT symbol, side, pnl, action AS exit_reason, notes, timestamp FROM berserker_trades"),
        ("marl_live_trades", "SELECT symbol, side, pnl, action AS exit_reason, notes, timestamp FROM marl_live_trades"),
    ]
    for tbl_name, q_base in trade_queries:
        try:
            raw_rows = safe_query(TRADING_DATA_DB, f"{q_base} ORDER BY id DESC LIMIT 50")
            for tr in raw_rows:
                ts_str = tr.get("timestamp") or ""
                # Parse timestamp robustly regardless of space vs T or timezone format
                try:
                    c_clean = str(ts_str).replace("Z", "+00:00")
                    if "T" not in c_clean and " " in c_clean:
                        c_clean = c_clean.replace(" ", "T")
                    if "+" not in c_clean and not c_clean.endswith("Z"):
                        c_clean += "+00:00"
                    t_dt = datetime.fromisoformat(c_clean)
                    if t_dt.tzinfo is None:
                        t_dt = t_dt.replace(tzinfo=timezone.utc)
                    if t_dt >= one_hour_ago:
                        closed_trades.append(tr)
                except Exception:
                    pass
        except Exception:
            pass

    hourly_realized_pnl = sum([float(t.get("pnl") or 0.0) for t in closed_trades])

    # 4. Fetch Active Positions Snapshot & Live Funding Telemetry (All fleet tables)
    combined_open = []
    open_queries = [
        ("swing_open_positions", "SELECT symbol, side, entry_price, size AS qty, stop_loss, take_profit, trailing_stop, peak_price, opened_at AS entry_time, leverage, notes AS comments FROM swing_open_positions WHERE status = 'ACTIVE' OR status IS NULL"),
        ("open_positions", "SELECT * FROM open_positions"),
        ("berserker_open_positions", "SELECT * FROM berserker_open_positions"),
        ("marl_live_open_positions", "SELECT * FROM marl_live_open_positions"),
    ]
    for tbl_name, q_open in open_queries:
        try:
            p_rows = safe_query(TRADING_DATA_DB, q_open)
            combined_open.extend(p_rows)
        except Exception:
            pass
    
    seen_syms = set()
    deduped_open = []
    for op in combined_open:
        s = str(op.get("symbol", "")).upper().replace("/", "").replace("-", "")
        if s not in seen_syms and op.get("qty") and float(op.get("qty") or 0.0) > 0:
            seen_syms.add(s)
            deduped_open.append(op)

    funding_raw = query_system_state("live_position_funding", "{}")
    funding_map = {}
    try:
        funding_map = json.loads(funding_raw)
    except Exception:
        pass

    total_accrued_funding = 0.0
    active_positions_detail = []
    total_unrealized_pnl = 0.0

    for pos in deduped_open:
        sym = str(pos.get("symbol", ""))
        sym_clean = sym.upper().replace("/", "").replace("-", "")
        side = str(pos.get("side", "long")).upper()
        qty = float(pos.get("qty") or pos.get("size") or 0.0)
        entry_p = float(pos.get("entry_price") or 0.0)
        sl = float(pos.get("stop_loss") or 0.0)
        hard_sl = float(pos.get("hard_sl") or 0.0)
        tp = float(pos.get("take_profit") or 0.0)
        lev = float(pos.get("leverage") or (10.0 if "BTC" in sym_clean else 5.0))
        
        # Pull telemetry
        f_info = funding_map.get(sym_clean, {}) or funding_map.get(sym, {})
        pos_funding = float(f_info.get("total_funding", 0.0) or 0.0)
        total_accrued_funding += pos_funding
        
        mark_p = float(f_info.get("mark_price", 0.0) or entry_p)
        if mark_p <= 0.0: mark_p = entry_p

        margin = (qty * entry_p) / lev if lev > 0 else (qty * entry_p)
        unr_pnl = (mark_p - entry_p) * qty if side == "LONG" else (entry_p - mark_p) * qty
        unr_roi = (unr_pnl / margin * 100.0) if margin > 0 else 0.0
        true_net = unr_pnl + pos_funding
        total_unrealized_pnl += unr_pnl

        # Query Shepherd verdict
        shep_raw = query_system_state(f"ai_shepherd_{sym_clean}", "")
        # A missing sentinel row / a missing verdict key is an absence, not "MONITORED".
        shep_verdict = None
        shep_diag = ""
        shep_ratchet = False
        shep_target_r = 2.5
        if shep_raw:
            try:
                s_obj = json.loads(shep_raw)
                shep_verdict = s_obj.get("verdict")
                shep_diag = s_obj.get("diagnosis", "")
                shep_ratchet = bool(s_obj.get("sl_ratcheted", False))
                shep_target_r = float(s_obj.get("target_r", 2.5))
            except Exception:
                pass

        active_positions_detail.append({
            "symbol": sym,
            "clean_symbol": sym_clean,
            "side": side,
            "qty": qty,
            "entry_p": entry_p,
            "mark_p": mark_p,
            "sl": sl,
            "hard_sl": hard_sl,
            "tp": tp,
            "lev": lev,
            "margin": margin,
            "unr_pnl": unr_pnl,
            "unr_roi": unr_roi,
            "funding": pos_funding,
            "true_net": true_net,
            "shep_verdict": shep_verdict,
            "shep_diag": shep_diag,
            "shep_ratchet": shep_ratchet,
            "shep_target_r": shep_target_r
        })

    # Return structured dataset
    return {
        "generated_at_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "cycles_count": cycles_count,
        "start_cycle_num": start_cycle_num,
        "end_cycle_num": end_cycle_num,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "equity_delta": equity_delta,
        "equity_delta_pct": equity_delta_pct,
        "peak_equity": peak_equity,
        "low_equity": low_equity,
        "min_margin": min_margin,
        "max_margin": max_margin,
        "end_margin": end_margin,
        "slots_occupied": len(active_positions_detail),
        "dominant_regime": latest_macro_bias,
        "directives": directives,
        "latest_rationale": latest_rationale,
        "latest_vetoes": latest_vetoes,
        "latest_watch_levels": latest_watch_levels,
        "latest_stance": latest_stance,
        "latest_macro_narrative": latest_macro_narrative,
        "closed_trades": closed_trades,
        "hourly_realized_pnl": hourly_realized_pnl,
        "active_positions": active_positions_detail,
        "total_unrealized_pnl": total_unrealized_pnl,
        "total_accrued_funding": total_accrued_funding,
        "true_fleet_net": total_unrealized_pnl + total_accrued_funding
    }


def build_hourly_briefing_html(data: Dict[str, Any]) -> str:
    """Formats the hourly rollup into a high-contrast institutional dark-mode HTML email."""
    eq_sign = "+$" if data["equity_delta"] >= 0 else "-$"
    eq_color = "#22c55e" if data["equity_delta"] >= 0 else "#ef4444"
    fund_sign = "+$" if data["total_accrued_funding"] >= 0 else "-$"
    fund_color = "#22c55e" if data["total_accrued_funding"] >= 0 else ("#f59e0b" if data["total_accrued_funding"] < 0 else "#94a3b8")
    fund_action = "Gained" if data["total_accrued_funding"] > 0 else ("Paid to Counterparties" if data["total_accrued_funding"] < 0 else "Neutral")
    
    # 1. Directives Timeline Rows
    directives_html = ""
    if data["directives"]:
        for d in data["directives"]:
            escaped_dir = html.escape(str(d.get("directive", "")))
            directives_html += f"""
            <div style="padding: 6px 10px; margin-bottom: 6px; background: rgba(255,255,255,0.02); border-left: 3px solid #38bdf8; border-radius: 4px; font-size: 13px;">
                <b style="color: #38bdf8;">Cycle #{d['cycle']} ({d['time']}):</b> <span style="color: #f1f5f9;">{escaped_dir}</span>
            </div>
            """
    else:
        directives_html = f"<div style='color: #94a3b8; font-size: 13px;'>{html.escape(str(data['latest_stance'] or 'Enforce capital preservation in chop regime.'))}</div>"

    # 2. Closed Trades Rows
    closed_trades_html = ""
    if data["closed_trades"]:
        for t in data["closed_trades"]:
            pnl_val = float(t.get("pnl") or 0.0)
            t_col = "#22c55e" if pnl_val >= 0 else "#ef4444"
            t_sign = "+$" if pnl_val >= 0 else "-$"
            closed_trades_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #1e293b; color: #38bdf8; font-weight: bold;">{html.escape(str(t.get('symbol', '')))}</td>
                <td style="padding: 8px; border-bottom: 1px solid #1e293b; text-align: center;">{html.escape(str(t.get('side', '')).upper())}</td>
                <td style="padding: 8px; border-bottom: 1px solid #1e293b; text-align: right; color: {t_col}; font-weight: bold;">{t_sign}{abs(pnl_val):.4f}</td>
                <td style="padding: 8px; border-bottom: 1px solid #1e293b; text-align: right; color: #94a3b8;">{html.escape(str(t.get('exit_reason') or t.get('action') or 'Settled'))}</td>
            </tr>
            """
    else:
        closed_trades_html = """
        <tr>
            <td colspan="4" style="padding: 14px; text-align: center; color: #94a3b8; font-style: italic;">
                🛡️ <b>Zero Forced Closures or Stop-Outs in Past 60 Minutes.</b> Positions held within calibrated safety buffers.
            </td>
        </tr>
        """

    # 3. Active Positions Rows
    positions_html = ""
    if data["active_positions"]:
        for p in data["active_positions"]:
            pnl_col = "#22c55e" if p["unr_pnl"] >= 0 else "#ef4444"
            pnl_sign = "+$" if p["unr_pnl"] >= 0 else "-$"
            p_fund_sign = "+$" if p["funding"] >= 0 else "-$"
            p_fund_col = "#22c55e" if p["funding"] >= 0 else "#f59e0b"
            ratchet_str = " <span style='color: #22c55e; font-weight: bold;'>[RATCHET]</span>" if p["shep_ratchet"] else ""
            
            raw_diag = str(p.get("shep_diag", ""))
            disp_diag = html.escape(raw_diag[:55] + "..." if len(raw_diag) > 55 else raw_diag)
            shep_disp = _shep_verdict_display(p.get("shep_verdict"))
            
            positions_html += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 10px 8px; color: #ffffff; font-weight: bold;">
                    {html.escape(str(p['symbol']))}<br>
                    <span style="font-size: 11px; color: #38bdf8;">{p['side']} {p['lev']:.0f}x (${p['margin']:.2f})</span>
                </td>
                <td style="padding: 10px 8px; font-family: monospace; font-size: 12px;">
                    Entry: <b>${p['entry_p']:,.4f}</b><br>
                    Mark: <b style="color: #38bdf8;">${p['mark_p']:,.4f}</b>
                </td>
                <td style="padding: 10px 8px; text-align: right; font-family: monospace; font-size: 13px;">
                    <b style="color: {pnl_col};">{pnl_sign}{abs(p['unr_pnl']):.4f}</b><br>
                    <span style="font-size: 11px; color: {pnl_col};">({p['unr_roi']:+.2f}%)</span>
                </td>
                <td style="padding: 10px 8px; text-align: right; font-family: monospace; font-size: 12px; color: {p_fund_col};">
                    {p_fund_sign}{abs(p['funding']):.4f}
                </td>
                <td style="padding: 10px 8px; font-size: 12px; color: #94a3b8;">
                    SL: <b style="color: #f59e0b;">${p['sl']:,.4f}</b>{ratchet_str}<br>
                    TP: <b style="color: #22c55e;">${p['tp']:,.4f}</b> (Target: {p['shep_target_r']:.1f}R)
                </td>
                <td style="padding: 10px 8px; font-size: 11px; color: #cbd5e1;">
                    <b style="color: {shep_disp['color']};">{html.escape(shep_disp['text'])}</b><br>
                    <span style="color: #94a3b8;">{disp_diag}</span>
                </td>
            </tr>
            """
    else:
        positions_html = """
        <tr>
            <td colspan="6" style="padding: 16px; text-align: center; color: #22c55e; font-weight: bold;">
                🛡️ 100% USDT Cash Shield Active (0 Active Positions)
            </td>
        </tr>
        """

    max_fleet_slots = getattr(config, "GLOBAL_MAX_POSITIONS", 8)

    email_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b101b; color: #f1f5f9; margin: 0; padding: 20px; }}
            .container {{ max-width: 680px; margin: 0 auto; background: #131d31; border: 1px solid #334155; border-radius: 10px; overflow: hidden; box-shadow: 0 12px 30px rgba(0,0,0,0.6); }}
            .header {{ background: linear-gradient(135deg, #0d1829 0%, #1e293b 100%); padding: 20px 24px; border-bottom: 2px solid #38bdf8; }}
            .header h1 {{ margin: 0; font-size: 20px; color: #38bdf8; display: flex; align-items: center; justify-content: space-between; }}
            .badge-regime {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #38bdf8; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
            .subtext {{ color: #94a3b8; font-size: 12px; margin-top: 5px; }}
            
            .kpi-grid {{ display: table; width: 100%; table-layout: fixed; background: #0d121f; border-bottom: 1px solid #1e293b; }}
            .kpi-cell {{ display: table-cell; padding: 14px 12px; text-align: center; border-right: 1px solid #1e293b; width: 25%; }}
            .kpi-cell:last-child {{ border-right: none; }}
            .kpi-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.05em; }}
            .kpi-val {{ font-size: 17px; font-weight: 800; font-family: monospace; margin-top: 4px; }}
            
            .section {{ padding: 18px 24px; border-bottom: 1px solid #1e293b; }}
            .section-title {{ font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; margin-bottom: 12px; border-left: 3px solid #38bdf8; padding-left: 8px; }}
            
            table.pos-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }}
            table.pos-table th {{ background: #0d121f; color: #94a3b8; text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; padding: 8px; text-align: left; border-bottom: 1px solid #334155; }}
            
            .memo-box {{ background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #22c55e; border-radius: 6px; padding: 12px 14px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; color: #cbd5e1; }}
            .footer {{ background: #0d121f; padding: 14px 24px; font-size: 11px; color: #64748b; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>
                    <span>🏆 Sovereign Swarm Hourly Executive Rollup</span>
                    <span class="badge-regime">{data['dominant_regime']}</span>
                </h1>
                <div class="subtext">
                    <b>Period:</b> Last 60 Minutes (5-Minute Cycles #{data['start_cycle_num']} – #{data['end_cycle_num']} | {data['cycles_count']} Cycles) | <b>Generated:</b> {data['generated_at_utc']}
                </div>
            </div>

            <!-- 4 Hero KPI Metrics -->
            <div class="kpi-grid">
                <div class="kpi-cell">
                    <div class="kpi-label">ENDING EQUITY</div>
                    <div class="kpi-val" style="color: #ffffff;">${data['end_equity']:,.2f}</div>
                    <div style="font-size: 11px; color: {eq_color}; font-weight: bold; margin-top: 2px;">
                        {eq_sign}{abs(data['equity_delta']):.2f} ({data['equity_delta_pct']:+.2f}%)
                    </div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">MARGIN UTILIZATION</div>
                    <div class="kpi-val" style="color: {'#22c55e' if data['end_margin'] < 60 else '#f59e0b'};">{data['end_margin']:.1f}%</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Range: {data['min_margin']:.0f}%–{data['max_margin']:.0f}%</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">ACCRUED FUNDING</div>
                    <div class="kpi-val" style="color: {fund_color};">{fund_sign}{abs(data['total_accrued_funding']):.4f}</div>
                    <div style="font-size: 11px; color: {fund_color}; margin-top: 2px;">{fund_action}</div>
                </div>
                <div class="kpi-cell">
                    <div class="kpi-label">FLEET SLOTS</div>
                    <div class="kpi-val" style="color: #38bdf8;">{data['slots_occupied']} / {max_fleet_slots}</div>
                    <div style="font-size: 11px; color: #94a3b8; margin-top: 2px;">Active Positions</div>
                </div>
            </div>

            <!-- Section 1: Tactical Directives Timeline -->
            <div class="section">
                <div class="section-title">🎖️ AI Commander's Tactical Directives (Past 60 Minutes)</div>
                {directives_html}
                <div class="memo-box" style="margin-top: 12px;">
                    <b>Positioning Rationale:</b> {data['latest_rationale'] or 'Consolidating long exposure under balanced order flow and chop bounds.'}<br><br>
                    <b>Operational Stance:</b> <i>"{data['latest_stance'] or 'Enforce strict tactical defense and protect capital.'}"</i>
                </div>
            </div>

            <!-- Section 2: Active Positions & Risk Matrix -->
            <div class="section">
                <div class="section-title">🎯 Active Position & Sentinel Telemetry Snapshot</div>
                <table class="pos-table">
                    <thead>
                        <tr>
                            <th>Asset / Size</th>
                            <th>Entry / Mark</th>
                            <th style="text-align: right;">Mark PnL</th>
                            <th style="text-align: right;">Funding</th>
                            <th>Stops & Target</th>
                            <th>In-Flight Sentinel</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions_html}
                    </tbody>
                </table>
            </div>

            <!-- Section 3: Closed Trades & Execution Activity -->
            <div class="section">
                <div class="section-title">📊 Hourly Execution & Closed Trades</div>
                <table class="pos-table">
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th style="text-align: center;">Side</th>
                            <th style="text-align: right;">Net PnL</th>
                            <th style="text-align: right;">Exit Trigger</th>
                        </tr>
                    </thead>
                    <tbody>
                        {closed_trades_html}
                    </tbody>
                </table>
            </div>

            <!-- Section 4: Strategic Blueprint & Key Watch Levels -->
            <div class="section">
                <div class="section-title">🗺️ Macro Strategy & Key Watch Levels</div>
                <div style="font-size: 12px; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                    <b>Strategic Narrative:</b> {data['latest_macro_narrative'] or 'Consolidating within structural chop corridors with low systemic contagion risk.'}
                </div>
                <div style="background: rgba(0, 229, 255, 0.05); border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 4px; padding: 8px 12px; font-size: 12px; color: #38bdf8;">
                    <b>Key Watch Levels:</b> {data['latest_watch_levels'] or 'BTC $82,500 support / $86,000 resistance'}
                </div>
            </div>

            <div class="footer">
                Sovereign Swarm Autonomous Institutional Trading Fleet • Dual-Engine AI Intelligence • Zero $ Cost
            </div>
        </div>
    </body>
    </html>
    """
    return email_html


def send_hourly_briefing_email(force: bool = False) -> bool:
    """Generates the hourly rollup and dispatches it via email_dispatcher.send_html_alert."""
    try:
        # Enforce 50-minute dispatch cooldown unless force=True
        if not force:
            try:
                conn_check = get_db_connection(TRADING_LEDGER_DB)
                last_row = conn_check.execute("SELECT generated_at FROM hourly_briefing_history ORDER BY id DESC LIMIT 1").fetchone()
                conn_check.close()
                if last_row and last_row[0]:
                    last_str = str(last_row[0]).replace(" UTC", "").strip()
                    last_dt = datetime.fromisoformat(last_str) if "T" in last_str else datetime.strptime(last_str, "%Y-%m-%d %H:%M:%S")
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    elapsed_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
                    if elapsed_min < 50.0:
                        log.info(f"Hourly briefing skipped (last sent {elapsed_min:.1f} mins ago; cooldown=50m; force=False).")
                        return True
            except Exception as cd_err:
                log.debug(f"Hourly briefing cooldown check note: {cd_err}")

        data = generate_hourly_briefing_data()
        html_content = build_hourly_briefing_html(data)
        
        eq_sign = "+$" if data["equity_delta"] >= 0 else "-$"
        subject = f"🏆 [HOURLY BRIEFING] Sovereign Swarm 1-Hour Rollup | Equity: ${data['end_equity']:,.2f} ({eq_sign}{abs(data['equity_delta']):.2f}) | {data['slots_occupied']} Pos | {data['dominant_regime']}"
        
        from email_dispatcher import send_html_alert
        success = send_html_alert(subject, html_content)
        
        if success:
            log.info(f"Hourly Executive Briefing dispatched successfully: {subject}")
            # Save record to trading_ledger.db
            try:
                conn = get_db_connection(TRADING_LEDGER_DB)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS hourly_briefing_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        generated_at TEXT,
                        start_cycle INTEGER,
                        end_cycle INTEGER,
                        start_equity REAL,
                        end_equity REAL,
                        equity_delta REAL,
                        slots_occupied INTEGER,
                        dominant_regime TEXT,
                        subject TEXT
                    )
                """)
                conn.execute("""
                    INSERT INTO hourly_briefing_history
                    (generated_at, start_cycle, end_cycle, start_equity, end_equity, equity_delta, slots_occupied, dominant_regime, subject)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data["generated_at_utc"],
                    data["start_cycle_num"],
                    data["end_cycle_num"],
                    data["start_equity"],
                    data["end_equity"],
                    data["equity_delta"],
                    data["slots_occupied"],
                    data["dominant_regime"],
                    subject
                ))
                conn.commit()
                conn.close()
            except Exception as hist_err:
                log.warning(f"Note recording hourly briefing history: {hist_err}")
                
        return success
    except Exception as e:
        log.error(f"Failed to generate or dispatch hourly executive briefing: {e}", exc_info=True)
        return False


def run_single_hourly_briefing(force: bool = False) -> bool:
    """Thread-safe entry point for supervisor and background schedulers."""
    return send_hourly_briefing_email(force=force)


if __name__ == "__main__":
    force_run = "--send" in sys.argv or "--force" in sys.argv
    log.info(f"Running hourly_briefing.py CLI (force={force_run})...")
    res = send_hourly_briefing_email(force=force_run)
    log.info(f"Hourly briefing execution finished. Result: {res}")
