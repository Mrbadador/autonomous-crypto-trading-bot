# morning_briefing.py — Sovereign Swarm Morning Diagnostic
# Generates morning_briefing.txt in the root workspace directory.
# Designed to run headless at 06:15 local time via Windows Task Scheduler.

import os
import sys
import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager

# ── Encoding safety (headless Task Scheduler has no UTF-8 console) ─────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MORNING_BRIEFING: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("morning_briefing")

# ── Absolute paths ─────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
TRADING_DATA_DB  = os.path.join(BASE_DIR, "trading_data.db")
NIGHTLY_LOG      = os.path.join(BASE_DIR, "logs", "nightly_master.log")
POLICY_MODEL     = os.path.join(BASE_DIR, "policy_model.pth")
MEMORY_JSON      = os.path.join(BASE_DIR, "learned_memory.json")
BRIEFING_PATH    = os.path.join(BASE_DIR, "morning_briefing.txt")
LEDGER_DB        = os.path.join(BASE_DIR, "trading_ledger.db")


# ── DB helper ──────────────────────────────────────────────────────────────────
@contextmanager
def get_db(path: str, timeout: float = 8.0):
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=8000;")
        yield conn
    finally:
        conn.close()


def safe_query(db_path: str, sql: str, params=()) -> list[dict]:
    """Run a query and return rows as dicts, or [] on any error."""
    if not os.path.exists(db_path):
        return []
    try:
        with get_db(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"Query failed on {os.path.basename(db_path)}: {e}")
        return []


# ── Section builders ───────────────────────────────────────────────────────────
def section(title: str) -> str:
    bar = "=" * 60
    return f"\n{bar}\n  {title}\n{bar}\n"


# ── 1. Midnight Pivot Check ────────────────────────────────────────────────────
def check_midnight_pivot() -> str:
    lines = [section("1. MIDNIGHT PIVOT — 00:05 UTC POLICY UPDATE")]

    # policy_model.pth timestamp
    if os.path.exists(POLICY_MODEL):
        mtime = datetime.fromtimestamp(os.path.getmtime(POLICY_MODEL), tz=timezone.utc)
        age_h = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        lines.append(f"  policy_model.pth last modified : {mtime.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"  File age                       : {age_h:.1f} hours ago")
        lines.append(f"  File size                      : {os.path.getsize(POLICY_MODEL):,} bytes")

        # Did the update happen within the last 8 hours (covers midnight window)?
        if age_h <= 8:
            lines.append("  Midnight Pivot Status          : [SUCCESS] Weights updated cleanly.")
        else:
            lines.append("  Midnight Pivot Status          : [STALE] Model not updated in 8+ hours!")
    else:
        lines.append("  [ERROR] policy_model.pth NOT FOUND — trainer may have crashed!")

    # Scan nightly_master.log for last pivot block
    if os.path.exists(NIGHTLY_LOG):
        with open(NIGHTLY_LOG, "r", encoding="utf-8", errors="ignore") as f:
            log_lines = f.readlines()
        # Extract last 20 lines of the log for context
        tail = [l.rstrip() for l in log_lines[-20:]]

        # Look for terminal markers
        success = any("NIGHTLY SEQUENCE COMPLETE" in l for l in tail)
        error   = any("error (200)" in l.lower() or "forrtl" in l.lower() for l in log_lines[-50:])

        lines.append(f"\n  --- nightly_master.log (last 20 lines) ---")
        lines.extend(f"  {l}" for l in tail)
        lines.append(f"\n  Log outcome   : {'[SUCCESS] Sequence completed cleanly.' if success else '[WARN] Completion marker not found in tail.'}")
        lines.append(f"  Fortran crash : {'[DETECTED] error (200) found in recent log!' if error else '[CLEAN] No Fortran crash detected.'}")
    else:
        lines.append(f"\n  [WARN] nightly_master.log not found at {NIGHTLY_LOG}")

    # Latest audit session from learned_memory.json
    if os.path.exists(MEMORY_JSON):
        try:
            with open(MEMORY_JSON, "r", encoding="utf-8") as f:
                mem = json.load(f)
            audit_keys = sorted(k for k in mem if k.startswith("audit_"))
            if audit_keys:
                last = mem[audit_keys[-1]]
                m    = last.get("metrics", {})
                lines.append(f"\n  Last Audit Session : {audit_keys[-1]}")
                lines.append(f"  Win Rate           : {m.get('win_rate', 0)*100:.1f}%")
                lines.append(f"  Total PnL          : {m.get('total_pnl', 0):+.4f} USDT")
                lines.append(f"  Total Reward       : {m.get('total_reward', 0):+.6f}")
                rules = last.get("rules", [])
                if rules:
                    lines.append(f"  Defensive Rules    : {len(rules)} rule(s) fired:")
                    for r in rules:
                        lines.append(f"    >> {r}")
                else:
                    lines.append(f"  Defensive Rules    : None (session healthy)")
        except Exception as e:
            lines.append(f"  [ERROR] Could not read learned_memory.json: {e}")

    return "\n".join(lines)


# ── 2. Live Portfolio & 24h PnL ────────────────────────────────────────────────
def check_portfolio() -> str:
    lines = [section("2. LIVE PORTFOLIO STATUS & 24H PNL")]

    # Open positions (MARL layer)
    open_pos = safe_query(
        TRADING_DATA_DB,
        "SELECT * FROM open_positions UNION ALL SELECT * FROM berserker_open_positions UNION ALL SELECT * FROM marl_live_open_positions ORDER BY entry_time ASC"
    )
    if open_pos:
        lines.append(f"  Open Swarm Positions : {len(open_pos)}")
        for p in open_pos:
            lines.append(
                f"    [{p.get('side','?').upper()}] {p.get('symbol','?')} | "
                f"Entry=${p.get('entry_price',0):,.2f}  "
                f"TP=${p.get('take_profit',0):,.2f}  "
                f"SL=${p.get('stop_loss',0):,.2f}  "
                f"Qty={p.get('qty',0):.6f}  "
                f"Lev={p.get('leverage',1)}x | "
                f"Entered: {str(p.get('entry_time','?'))[:19]} UTC"
            )
    else:
        lines.append("  Open Swarm Positions : 0  (flat — no active exposure)")

    # 24-hour closed trades
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    closed = safe_query(
        TRADING_DATA_DB,
        "SELECT * FROM trades WHERE (status = 'closed' OR status = 'CLOSED') AND timestamp >= ? ORDER BY timestamp DESC",
        (cutoff,)
    )

    if closed:
        total_pnl = sum(float(t.get("pnl") or 0) for t in closed)
        wins      = sum(1 for t in closed if float(t.get("pnl") or 0) > 0)
        losses    = sum(1 for t in closed if float(t.get("pnl") or 0) < 0)
        win_rate  = (wins / len(closed) * 100) if closed else 0

        lines.append(f"\n  24h Closed Trades   : {len(closed)}")
        lines.append(f"  24h Win / Loss      : {wins}W / {losses}L  ({win_rate:.0f}% win rate)")
        lines.append(f"  24h Realized PnL    : {total_pnl:+.4f} USDT")

        lines.append(f"\n  Recent Closed Trades (last 5):")
        for t in closed[:5]:
            pnl_val = float(t.get("pnl") or 0)
            marker  = "WIN" if pnl_val > 0 else "LOSS"
            lines.append(
                f"    [{marker}] {t.get('symbol','?')} {str(t.get('side','?')).upper()} | "
                f"PnL={pnl_val:+.4f}  Type={t.get('type','?')}  "
                f"Regime={t.get('regime','?')} | {str(t.get('timestamp','?'))[:19]} UTC"
            )
    else:
        lines.append("\n  24h Closed Trades   : 0")

    # Latest telemetry snapshot
    latest_tick = safe_query(
        TRADING_DATA_DB,
        "SELECT symbol, timestamp, price, action, reasoning FROM telemetry "
        "ORDER BY timestamp DESC LIMIT 3"
    )
    if latest_tick:
        lines.append(f"\n  Latest Telemetry Snapshots:")
        for t in latest_tick:
            lines.append(
                f"    {t.get('symbol','?')} @ ${float(t.get('price',0)):,.2f}  "
                f"| Action={t.get('action','?')}  "
                f"| {str(t.get('timestamp','?'))[:19]}"
            )
            lines.append(f"      Reasoning: {str(t.get('reasoning','?'))[:80]}")

    return "\n".join(lines)


# ── 3. Ghost Trades Ledger ─────────────────────────────────────────────────────
def check_ghost_trades() -> str:
    lines = [section("3. GHOST TRADES — SNIPER REHABILITATION TRACKER")]

    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    all_ghosts = safe_query(
        TRADING_DATA_DB,
        "SELECT * FROM ghost_trades ORDER BY timestamp DESC"
    )
    new_ghosts = safe_query(
        TRADING_DATA_DB,
        "SELECT * FROM ghost_trades WHERE timestamp >= ? ORDER BY timestamp DESC",
        (cutoff_24h,)
    )

    lines.append(f"  Total Ghost Trades  : {len(all_ghosts)}")
    lines.append(f"  New (last 24h)      : {len(new_ghosts)}")

    if all_ghosts:
        # Win rate
        resolved = [g for g in all_ghosts if g.get("outcome") is not None]
        wins     = [g for g in resolved if str(g.get("outcome","")).upper() == "WIN"]
        losses   = [g for g in resolved if str(g.get("outcome","")).upper() == "LOSS"]
        wr       = (len(wins) / len(resolved) * 100) if resolved else 0
        open_g   = [g for g in all_ghosts if g.get("exit_price") is None]

        lines.append(f"  Resolved Ghosts     : {len(resolved)}")
        lines.append(f"  Ghost Win Rate      : {wr:.0f}%  ({len(wins)}W / {len(losses)}L)")
        lines.append(f"  Currently Open      : {len(open_g)}")

        # Check paper_tp / paper_sl population
        has_tp_sl = sum(1 for g in all_ghosts if g.get("paper_tp") and g.get("paper_sl"))
        lines.append(f"  V32.4 TP/SL columns : {has_tp_sl}/{len(all_ghosts)} populated (look-ahead fix)")

        # Sniper gate recommendation
        if len(resolved) > 0:
            if wr >= 50:
                lines.append(f"\n  Sniper Gate         : [CONSIDER LIVE] Ghost win rate >= 50%")
            else:
                lines.append(f"\n  Sniper Gate         : [GHOST-ONLY] Win rate below 50% — remain in rehab mode")

        # List all ghost trades
        lines.append(f"\n  --- All Ghost Trade Entries ---")
        for g in all_ghosts:
            status = "OPEN" if g.get("exit_price") is None else f"CLOSED ({g.get('outcome','?')})"
            tp_sl  = (f"TP=${g.get('paper_tp',0):,.2f}  SL=${g.get('paper_sl',0):,.2f}"
                      if g.get("paper_tp") else "TP/SL: NOT SET (pre-V32.4)")
            lines.append(
                f"    ID={g.get('id','?')} | {g.get('symbol','?')} {str(g.get('signal','?')).upper()} "
                f"@ ${g.get('entry_price',0):,.2f} | {status} | {tp_sl}"
            )
            lines.append(f"      Entered: {str(g.get('timestamp','?'))[:19]} UTC")
    else:
        lines.append("\n  No ghost trades found in database.")

    return "\n".join(lines)


# ── Master Report Builder ──────────────────────────────────────────────────────
def generate_briefing() -> str:
    now_utc   = datetime.now(timezone.utc)
    now_local = datetime.now()

    header = (
        f"{'*' * 60}\n"
        f"  SOVEREIGN SWARM — MORNING BRIEFING\n"
        f"  Generated : {now_local.strftime('%Y-%m-%d %H:%M:%S')} Local\n"
        f"             ({now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)\n"
        f"{'*' * 60}"
    )
    footer = (
        f"\n{'=' * 60}\n"
        f"  END OF BRIEFING — Have a great session!\n"
        f"{'=' * 60}\n"
    )

    log.info("Building Morning Briefing...")
    pivot_section   = check_midnight_pivot()
    log.info("Section 1 complete.")
    port_section    = check_portfolio()
    log.info("Section 2 complete.")
    ghost_section   = check_ghost_trades()
    log.info("Section 3 complete.")

    return "\n".join([header, pivot_section, port_section, ghost_section, footer])


def send_morning_briefing_email(briefing_text: str):
    """Formats and dispatches the Morning Briefing as an institutional HTML email."""
    try:
        from email_dispatcher import send_html_alert
        now_str = datetime.now().strftime("%Y-%m-%d")
        subject = f"🌅 [MORNING BRIEFING] Sovereign Swarm Daily Report — {now_str}"
        
        # Convert plain text sections to styled HTML pre block
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b101b; color: #f1f5f9; padding: 20px; }}
                .container {{ max-width: 680px; background: #131d31; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
                h2 {{ color: #38bdf8; margin-top: 0; font-size: 20px; border-bottom: 2px solid #1e293b; padding-bottom: 12px; text-align: center; }}
                pre {{ background: #080d1a; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; color: #e2e8f0; white-space: pre-wrap; word-break: break-word; line-height: 1.5; }}
                .footer {{ font-size: 11px; color: #475569; text-align: center; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>🌅 SOVEREIGN SWARM — MORNING BRIEFING</h2>
                <pre>{briefing_text}</pre>
                <div class="footer">
                    Generated automatically by the Sovereign Swarm Morning Diagnostic Engine.
                </div>
            </div>
        </body>
        </html>
        """
        send_html_alert(subject, html_content)
        log.info("Morning briefing email successfully dispatched.")
    except Exception as em_err:
        log.warning(f"Failed to dispatch morning briefing email: {em_err}")

# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    briefing = generate_briefing()

    # Write to file
    try:
        with open(BRIEFING_PATH, "w", encoding="utf-8") as f:
            f.write(briefing)
        log.info(f"Morning briefing saved to: {BRIEFING_PATH}")
    except IOError as e:
        log.error(f"Could not write briefing file: {e}")

    # Dispatch email
    send_morning_briefing_email(briefing)

    # Also print to console
    print(briefing)
    sys.exit(0)
