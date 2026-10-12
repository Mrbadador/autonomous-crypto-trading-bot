# email_notifier.py — Sovereign Swarm 6-Hour Heartbeat Reporter
# =============================================================
# Generates and dispatches a rich HTML telemetry email every 6 hours.
# Triggered by antigravity_agent_manager.py's async loop.
#
# Data sources  : trading_data.db, trading_ledger.db, logs/main.log,
#                 berserker_trader.BLACKLIST (imported live)
# SMTP transport: reuses email_dispatcher.send_html_alert() which reads
#                 GMAIL_USER / GMAIL_APP_PASSWORD from .env — no new creds.
#
# Schedule      : 00:00, 06:00, 12:00, 18:00 UTC
# Safe to call  : synchronously or via asyncio.to_thread()

import os
import sys
import sqlite3
import logging
import importlib
import importlib.util
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

log = logging.getLogger("email_notifier")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRADING_DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
MAIN_LOG_PATH   = os.path.join(BASE_DIR, "logs", "main.log")

# Heartbeat window: last 6 hours
WINDOW_HOURS = 6

# Sniper rehabilitation gate threshold
REHAB_TARGET_WR = 45.0  # percent


# ── DB helpers ─────────────────────────────────────────────────────────────────
@contextmanager
def _db(path: str, timeout: float = 8.0):
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=8000;")
    try:
        yield conn
    finally:
        conn.close()


def _safe_query(db_path: str, sql: str, params=()) -> list[dict]:
    if not os.path.exists(db_path):
        return []
    try:
        with _db(db_path) as c:
            return [dict(r) for r in c.execute(sql, params).fetchall()]
    except Exception as e:
        log.warning(f"DB query failed ({os.path.basename(db_path)}): {e}")
        return []


# ── Section builders ───────────────────────────────────────────────────────────

def _section_uptime() -> tuple[str, list[str]]:
    """Swarm uptime, daemon PID, WebSocket, heartbeat file check."""
    alerts: list[str] = []

    # Port 9812 check via socket connection
    ws_status = "UNKNOWN"
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            result_code = s.connect_ex(('127.0.0.1', 9812))
            if result_code == 0:
                ws_status = "LISTENING"
            else:
                ws_status = "DOWN"
                alerts.append("CRITICAL: WebSocket port 9812 is NOT listening.")
    except Exception as e:
        ws_status = f"CHECK FAILED ({e})"
        alerts.append(f"WARNING: Could not verify WebSocket port: {e}")

    # main.log heartbeat freshness
    log_status = "UNKNOWN"
    log_age_str = "N/A"
    if os.path.exists(MAIN_LOG_PATH):
        size  = os.path.getsize(MAIN_LOG_PATH)
        mtime = datetime.fromtimestamp(os.path.getmtime(MAIN_LOG_PATH), tz=timezone.utc)
        age_m = (datetime.now(timezone.utc) - mtime).total_seconds() / 60
        log_age_str = f"{age_m:.0f} min ago ({mtime.strftime('%H:%M UTC')})"
        if size == 0:
            log_status = "EMPTY"
            alerts.append("WARNING: main.log is 0 bytes — heartbeat stream may be severed.")
        elif age_m > 30:
            log_status = f"STALE ({age_m:.0f} min)"
            alerts.append(f"WARNING: main.log last write was {age_m:.0f} minutes ago.")
        else:
            log_status = "LIVE"
    else:
        log_status = "MISSING"
        alerts.append("CRITICAL: main.log does not exist.")

    # Python process count (proxy for swarm health)
    proc_count = 0
    try:
        import subprocess
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=5
        )
        proc_count = max(0, r.stdout.count("python.exe") - 1)  # subtract header
    except Exception:
        pass

    rows = [
        ("WebSocket Port 9812",  ws_status,     "color:#10b981" if ws_status == "LISTENING" else "color:#ef4444"),
        ("main.log Status",      log_status,    "color:#10b981" if log_status == "LIVE"      else "color:#f59e0b"),
        ("main.log Last Write",  log_age_str,   ""),
        ("Active Python Processes", str(proc_count), "color:#38bdf8"),
        ("Report Generated",     datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), ""),
    ]
    return _table_html("1. Swarm Uptime & Daemon Health", rows), alerts


def _section_financials() -> tuple[str, list[str]]:
    """6-hour PnL and open positions."""
    alerts: list[str] = []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat()

    # Closed trades in window
    closed = _safe_query(
        TRADING_DATA_DB,
        "SELECT symbol, side, pnl, type, regime, timestamp FROM trades "
        "WHERE status='closed' AND timestamp >= ? ORDER BY timestamp DESC",
        (cutoff,)
    )
    total_pnl = sum(float(t.get("pnl") or 0) for t in closed)
    wins      = sum(1 for t in closed if float(t.get("pnl") or 0) > 0)
    losses    = sum(1 for t in closed if float(t.get("pnl") or 0) < 0)
    wr        = (wins / len(closed) * 100) if closed else 0

    pnl_color = "#10b981" if total_pnl >= 0 else "#ef4444"
    if total_pnl < -10:
        alerts.append(f"ALERT: 6h Net PnL is {total_pnl:+.2f} USDT — drawdown exceeds $10.")

    # Open positions across all fleet engines (Sniper, Berserker, MARL, Swing Architect)
    open_pos = _safe_query(
        TRADING_DATA_DB,
        "SELECT symbol, side, entry_price, take_profit, stop_loss, qty, leverage, entry_time FROM open_positions "
        "UNION ALL "
        "SELECT symbol, side, entry_price, take_profit, stop_loss, qty, leverage, entry_time FROM berserker_open_positions "
        "UNION ALL "
        "SELECT symbol, side, entry_price, take_profit, stop_loss, qty, leverage, entry_time FROM marl_live_open_positions "
        "UNION ALL "
        "SELECT symbol, side, entry_price, take_profit, stop_loss, size AS qty, leverage, opened_at AS entry_time FROM swing_open_positions "
        "ORDER BY entry_time ASC"
    )

    # Fetch live account balance from system_state
    bal_rows = _safe_query(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = 'balance'")
    live_bal = float(bal_rows[0]['value']) if bal_rows and bal_rows[0].get('value') and bal_rows[0]['value'] != 'None' else float(os.getenv("INITIAL_BALANCE", "250.00"))
    dep_rows = _safe_query(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = 'user_total_deposit'")
    init_bal = float(dep_rows[0]['value']) if dep_rows and dep_rows[0].get('value') else float(os.getenv("INITIAL_BALANCE", "250.00"))
    tot_roi = ((live_bal - init_bal) / init_bal) * 100.0 if init_bal > 0 else 0.0
    tot_profit = live_bal - init_bal
    
    max_slots_row = _safe_query(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = 'GLOBAL_MAX_POSITIONS'")
    max_slots = int(max_slots_row[0]['value']) if max_slots_row and max_slots_row[0].get('value') else 5

    rows = [
        ("Live Account Equity",      f"${live_bal:,.2f} USDT",         "color:#38bdf8;font-weight:bold"),
        ("Cumulative Account ROI",   f"{'+$' if tot_profit>=0 else '-$'}{abs(tot_profit):.2f} ({tot_roi:+.2f}%)", "color:#10b981;font-weight:bold"),
        ("6h Window",                f"Last {WINDOW_HOURS}h",         ""),
        ("Closed Trades (6h)",       str(len(closed)),                 ""),
        ("Win / Loss (6h)",          f"{wins}W / {losses}L",          ""),
        ("6h Win Rate",              f"{wr:.0f}%",                     "color:#10b981" if wr >= 50 else "color:#f59e0b"),
        ("6h Net Realized PnL",      f"${total_pnl:+.4f} USDT",       f"color:{pnl_color};font-weight:bold"),
        ("Active Coin Slots",        f"{len(open_pos)} / {max_slots}", "color:#38bdf8"),
    ]

    html = _table_html("2. Financials — Live Equity & 6H Performance", rows)

    # Open position detail table
    if open_pos:
        pos_rows = ""
        for p in open_pos:
            side_color = "#ef4444" if str(p.get("side","")).upper() == "SHORT" else "#10b981"
            pos_rows += (
                f"<tr>"
                f"<td class='value' style='text-align:left;color:#818cf8'>{p.get('symbol','?')}</td>"
                f"<td class='value' style='text-align:center;color:{side_color}'>{str(p.get('side','?')).upper()}</td>"
                f"<td class='value' style='text-align:right'>${float(p.get('entry_price',0)):,.2f}</td>"
                f"<td class='value' style='text-align:right'>${float(p.get('take_profit',0)):,.2f}</td>"
                f"<td class='value' style='text-align:right'>${float(p.get('stop_loss',0)):,.2f}</td>"
                f"<td class='value' style='text-align:center'>{p.get('leverage',1)}x</td>"
                f"</tr>"
            )
        html += f"""
        <h3 style='color:#cbd5e1;font-size:14px;margin-top:20px;text-transform:uppercase;
                   letter-spacing:.05em;border-left:3px solid #38bdf8;padding-left:8px;'>
            Open Position Registry
        </h3>
        <table width='100%' style='border-collapse:collapse;margin-top:8px;'>
            <thead>
                <tr>
                    <th style='text-align:left;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>Symbol</th>
                    <th style='text-align:center;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>Side</th>
                    <th style='text-align:right;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>Entry</th>
                    <th style='text-align:right;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>TP</th>
                    <th style='text-align:right;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>SL</th>
                    <th style='text-align:center;padding:6px 8px;color:#94a3b8;font-size:13px;border-bottom:1px solid #334155'>Lev</th>
                </tr>
            </thead>
            <tbody>{pos_rows}</tbody>
        </table>"""
    else:
        html += "<p style='color:#94a3b8;font-style:italic;margin-top:8px;font-size:13px;'>No open positions &mdash; fleet is flat.</p>"

    return html, alerts


def _section_rehabilitation() -> tuple[str, list[str]]:
    """Sniper ghost-trade rehabilitation metrics."""
    alerts: list[str] = []

    ghosts    = _safe_query(TRADING_DATA_DB, "SELECT * FROM ghost_trades ORDER BY timestamp DESC")
    resolved  = [g for g in ghosts if g.get("exit_price") is not None]
    wins_g    = [g for g in resolved if str(g.get("outcome","")).upper() == "WIN"]
    losses_g  = [g for g in resolved if str(g.get("outcome","")).upper() == "LOSS"]
    open_g    = [g for g in ghosts if g.get("exit_price") is None]
    wr        = (len(wins_g) / len(resolved) * 100) if resolved else 0
    tp_sl_ok  = all(g.get("paper_tp") is not None and g.get("paper_sl") is not None for g in ghosts) if ghosts else True

    gap_to_gate = REHAB_TARGET_WR - wr
    if wr < REHAB_TARGET_WR:
        gate_status = f"GHOST-ONLY ({gap_to_gate:.0f}% below gate)"
        gate_color  = "#f59e0b"
    else:
        gate_status = "READY FOR LIVE — Win Rate >= 45%"
        gate_color  = "#10b981"
        alerts.append("NOTICE: Sniper ghost win rate >= 45%. Consider opening the live gate.")

    if not tp_sl_ok:
        alerts.append("WARNING: Some ghost trades are missing paper_tp/paper_sl entries.")

    rows = [
        ("Total Ghost Trades",     str(len(ghosts)),       ""),
        ("Resolved / Open",        f"{len(resolved)} / {len(open_g)}", ""),
        ("Ghost Wins / Losses",    f"{len(wins_g)}W / {len(losses_g)}L", ""),
        ("Simulated Win Rate",     f"{wr:.0f}%",            f"color:{gate_color}"),
        ("Target (Live Gate)",     f"{REHAB_TARGET_WR:.0f}%", "color:#94a3b8"),
        ("Gate Status",            gate_status,             f"color:{gate_color};font-weight:bold"),
        ("TP/SL Populated",        "YES (all)" if tp_sl_ok else "NO — MISSING ENTRIES",
                                   "color:#10b981" if tp_sl_ok else "color:#ef4444"),
    ]
    return _table_html("3. Sniper Rehabilitation Tracker", rows), alerts


def _section_near_misses() -> tuple[str, list[str]]:
    """Near-miss / candidate trades filtered out by AI & Quant gates in past 6h."""
    alerts: list[str] = []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat()
    rows = _safe_query(
        TRADING_DATA_DB,
        """SELECT timestamp, engine, message FROM logs 
           WHERE (
               message LIKE '%Strike Aborted%' 
               OR message LIKE '%VETO%' 
               OR message LIKE '%TRADE BLOCKED%' 
               OR message LIKE '%MACRO HALT%'
               OR message LIKE '%HURST-CHOP-GATE%'
               OR message LIKE '%NEWS-CALENDAR-VETO%'
               OR message LIKE '%VaR-HALT%'
               OR message LIKE '%CORE-PRIORITY-GATE%'
           )
           AND timestamp >= ?
           ORDER BY id DESC LIMIT 5""",
        (cutoff,)
    )
    if not rows:
        html = _section_header("4. Near-Miss Setups & Filtered Trades")
        html += "<p style='color:#94a3b8;font-style:italic;font-size:13px;margin-top:8px;'>No candidate strikes were vetoed in the past 6 hours.</p>"
        return html, alerts

    cards = ""
    for r in rows:
        ts = str(r.get("timestamp", ""))[:19].replace("T", " ")
        eng = str(r.get("engine", "SWARM")).upper()
        msg = str(r.get("message", ""))
        if "Strike Aborted" in msg:
            clean_msg = msg.split("Strike Aborted")[-1].lstrip(" (0.0):").strip()
        else:
            clean_msg = msg.strip()
        cards += f"""
        <div style='background:#1e293b;border-left:3px solid #f59e0b;border-radius:4px;padding:8px 12px;margin-bottom:8px;'>
            <div style='font-size:11px;color:#94a3b8;margin-bottom:3px;'>
                <span style='color:#f59e0b;font-weight:bold;'>⚡ {eng}</span> &bull; {ts} UTC
            </div>
            <div style='font-size:12px;color:#cbd5e1;line-height:1.4;'>{clean_msg}</div>
        </div>"""
    
    html = _section_header("4. Near-Miss Setups & Filtered Trades")
    html += f"<div style='margin-top:8px;'>{cards}</div>"
    return html, alerts


def _section_alerts(all_alerts: list[str], blacklist: list[str]) -> str:
    """Alerts and BLACKLIST section."""
    html  = _section_header("5. Alerts & Blacklist")

    if blacklist:
        bl_rows = "".join(
            f"<tr><td style='padding:6px 12px;border-bottom:1px solid #334155;"
            f"color:#ef4444;font-family:Consolas,monospace;font-size:13px'>{sym}</td>"
            f"<td style='padding:6px 12px;border-bottom:1px solid #334155;"
            f"color:#94a3b8;font-size:13px'>Hard quarantine — execution blocked</td></tr>"
            for sym in blacklist
        )
        html += f"""
        <table width='100%' style='border-collapse:collapse;margin-top:8px;margin-bottom:16px;'>
            <thead><tr>
                <th style='text-align:left;padding:6px 12px;color:#94a3b8;font-size:13px;border-bottom:1px solid #475569'>Quarantined Asset</th>
                <th style='text-align:left;padding:6px 12px;color:#94a3b8;font-size:13px;border-bottom:1px solid #475569'>Reason</th>
            </tr></thead>
            <tbody>{bl_rows}</tbody>
        </table>"""
    else:
        html += "<p style='color:#94a3b8;font-style:italic;font-size:13px;margin-top:8px;'>Blacklist is empty.</p>"

    if all_alerts:
        html += "<ul style='margin-top:12px;padding-left:20px;'>"
        for a in all_alerts:
            color = "#ef4444" if "CRITICAL" in a else "#f59e0b" if "WARNING" in a or "ALERT" in a else "#38bdf8"
            html += f"<li style='color:{color};font-size:13px;margin-bottom:6px'>{a}</li>"
        html += "</ul>"
    else:
        html += "<p style='color:#10b981;font-size:13px;margin-top:8px;'>No alerts. All systems nominal.</p>"

    return html


# ── HTML primitives ────────────────────────────────────────────────────────────
def _section_header(title: str) -> str:
    return (
        f"<h3 style='color:#cbd5e1;font-size:14px;margin-top:24px;text-transform:uppercase;"
        f"letter-spacing:.05em;border-left:3px solid #38bdf8;padding-left:8px;'>{title}</h3>"
    )


def _table_html(title: str, rows: list[tuple[str, str, str]]) -> str:
    header = _section_header(title)
    trows  = ""
    for label, value, style in rows:
        trows += (
            f"<tr>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #334155;"
            f"font-weight:600;color:#94a3b8;font-size:14px;width:50%'>{label}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #334155;"
            f"text-align:right;font-family:Consolas,monospace;font-size:14px;"
            f"color:#f1f5f9;{style}'>{value}</td>"
            f"</tr>"
        )
    return header + f"<table width='100%' style='border-collapse:collapse;margin-top:8px;'>{trows}</table>"


# ── Master report builder ──────────────────────────────────────────────────────
def build_report_html() -> tuple[str, str]:
    """
    Builds the full HTML report body and plain-text subject line.
    Returns (subject, html_body).
    """
    now_utc = datetime.now(timezone.utc)
    slot    = (now_utc.hour // 6) * 6  # 0, 6, 12, or 18

    # Collect all section data
    uptime_html,  alerts_uptime  = _section_uptime()
    finance_html, alerts_finance = _section_financials()
    rehab_html,   alerts_rehab   = _section_rehabilitation()
    near_html,    alerts_near    = _section_near_misses()
    all_alerts = alerts_uptime + alerts_finance + alerts_rehab + alerts_near

    # Read live BLACKLIST from berserker_trader without importing its heavy deps
    blacklist: list[str] = []
    try:
        with open(os.path.join(BASE_DIR, "berserker_trader.py"), "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        import ast, re
        # Find the BLACKLIST assignment block
        m = re.search(r"^BLACKLIST\s*=\s*(\[[\s\S]*?\n\])", content, re.MULTILINE)
        if m:
            # Strip inline comments so ast.literal_eval doesn't choke on '#' brackets
            clean = re.sub(r"#[^\n]*", "", m.group(1))
            blacklist = ast.literal_eval(clean)
    except Exception as e:
        log.warning(f"Could not read BLACKLIST from berserker_trader.py: {e}")

    if blacklist:
        all_alerts.insert(0, f"BLACKLIST ACTIVE: {len(blacklist)} asset(s) quarantined — {', '.join(blacklist)}")

    alerts_html = _section_alerts(all_alerts, blacklist)

    # Determine session emoji and label
    session_labels = {0: "00:00 UTC | Asian Open", 6: "06:00 UTC | London Open",
                      12: "12:00 UTC | NY Open", 18: "18:00 UTC | NY Close"}
    session_label = session_labels.get(slot, f"{slot:02d}:00 UTC")

    # PnL quick-read for subject line
    cutoff = (now_utc - timedelta(hours=WINDOW_HOURS)).isoformat()
    closed = _safe_query(
        TRADING_DATA_DB,
        "SELECT pnl FROM trades WHERE status='closed' AND timestamp >= ?", (cutoff,)
    )
    total_pnl = sum(float(t.get("pnl") or 0) for t in closed)
    alert_tag  = " [ALERT]" if any("CRITICAL" in a for a in all_alerts) else ""

    subject = (
        f"[SWARM 6H] {session_label} | PnL: ${total_pnl:+.2f} USDT"
        f" | Alerts: {len(all_alerts)}{alert_tag}"
    )

    html_body = f"""
    <html>
    <head>
        <meta charset='utf-8'>
        <style>
            body {{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
                   background:#0f172a;color:#f1f5f9;padding:20px;margin:0}}
            .container {{max-width:640px;background:#1e293b;border:1px solid #334155;
                         border-radius:12px;padding:28px;margin:0 auto;
                         box-shadow:0 10px 25px -3px rgba(0,0,0,.4)}}
            .header-bar {{background:linear-gradient(135deg,#1e3a5f,#0f2744);
                          border-radius:8px;padding:18px 20px;margin-bottom:20px;
                          border-left:4px solid #38bdf8}}
            .header-bar h2 {{margin:0;color:#38bdf8;font-size:20px;letter-spacing:.02em}}
            .header-bar p  {{margin:4px 0 0;color:#94a3b8;font-size:13px}}
            .footer {{font-size:11px;color:#475569;text-align:center;margin-top:28px;
                      border-top:1px solid #334155;padding-top:16px}}
        </style>
    </head>
    <body>
        <div class='container'>
            <div class='header-bar'>
                <h2>&#9889; SOVEREIGN SWARM &mdash; 6-HOUR HEARTBEAT</h2>
                <p>Session: {session_label} &nbsp;|&nbsp; Report: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>

            {uptime_html}
            {finance_html}
            {rehab_html}
            {near_html}
            {alerts_html}

            <div class='footer'>
                Automated telemetry dispatched by the Sovereign Swarm v32.6 &mdash; antigravity_agent_manager.py<br>
                Next report in approximately 6 hours.
            </div>
        </div>
    </body>
    </html>
    """
    return subject, html_body


# ── Public dispatch function ───────────────────────────────────────────────────
def send_6h_heartbeat() -> bool:
    """
    Build and dispatch the 6-hour heartbeat email.
    Uses the existing email_dispatcher transport (GMAIL_USER / GMAIL_APP_PASSWORD).
    Returns True on success, False on failure. Safe to call from asyncio.to_thread().
    """
    try:
        from email_dispatcher import send_html_alert
    except ImportError as e:
        log.error(f"[6H-HEARTBEAT] Cannot import email_dispatcher: {e}")
        return False

    try:
        subject, html_body = build_report_html()
        ok = send_html_alert(subject, html_body)
        if ok:
            log.info(f"[6H-HEARTBEAT] Dispatched successfully: {subject}")
        else:
            log.warning("[6H-HEARTBEAT] send_html_alert returned False (check GMAIL credentials).")
        return bool(ok)
    except Exception as e:
        log.error(f"[6H-HEARTBEAT] Unhandled error during report dispatch: {e}")
        return False


async def send_6h_heartbeat_async() -> bool:
    """Async wrapper — call from the agent manager's asyncio loop."""
    import asyncio
    return await asyncio.to_thread(send_6h_heartbeat)


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    log.info("Running email_notifier.py self-test (build + print, no email sent)...")
    subject, html = build_report_html()
    print(f"\nSUBJECT: {subject}\n")
    # Save preview
    preview_path = os.path.join(BASE_DIR, "logs", "heartbeat_preview.html")
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML preview saved to: {preview_path}")
    print("\nTo send a live test email, run: python -c \"from email_notifier import send_6h_heartbeat; send_6h_heartbeat()\"")
