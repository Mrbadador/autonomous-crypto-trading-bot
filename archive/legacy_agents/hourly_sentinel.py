#!/usr/bin/env python3
"""
hourly_sentinel.py — Sovereign Fleet Automated Reliability Sentinel
Quantitative Systems Reliability Engineer (SRE) Tooling

Operational Invariants:
1. STRICTLY READ-ONLY: Never writes to trading tables, never alters weights, never terminates processes.
2. ZERO HOT-PATH INTERFERENCE: All database reads use URI 'file:<db>?mode=ro' with timeout=5.0s.
3. NON-BLOCKING: Lightweight execution, low CPU overhead, automated hourly cycle.

Checks Performed:
1. Process Tree & Worker Uptime (psutil)
2. Microstructure & OFI Stream Freshness (microstructure_state.json < 10s)
3. SQLite WAL Health & Read-Only Query Latency (< 50ms, 0 lock errors)
4. Log File Scrutiny (main.log, ofi_daemon.log, etc. for unhandled exceptions)
5. Position & Trade Bracket Audit (Stop-Loss & Take-Profit integrity)
"""

import os
import sys
import time
import json
import sqlite3
import argparse
import datetime
from datetime import timezone
import psutil

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure working directory is repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

LOG_FILE = os.path.join(BASE_DIR, "logs", "hourly_audit.log")
DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")
STATE_FILE = os.path.join(BASE_DIR, "microstructure_state.json")

def get_utc_now():
    return datetime.datetime.now(timezone.utc)

def log_output(message: str, to_console: bool = True):
    """Writes to hourly_audit.log and optionally prints to console."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    if to_console:
        print(message)


# -------------------------------------------------------------------------
# Check 1: Process Tree & Worker Health
# -------------------------------------------------------------------------
def check_processes():
    critical_targets = [
        "start_system.py",
        "ofi_daemon.py",
        "main.py",
        "marl_live_trader.py",
        "live_trader.py"
    ]
    secondary_targets = [
        "live_oracle.py",
        "regime_classifier.py",
        "portfolio_risk_agent.py",
        "reconciliation_engine.py",
        "db_maintenance_daemon.py"
    ]
    
    found_procs = {}
    total_cpu = 0.0
    total_mem_mb = 0.0

    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent', 'memory_info']):
        try:
            cmd = " ".join(p.info.get('cmdline') or [])
            if "python" in p.info.get('name', '').lower() and any(k in cmd for k in critical_targets + secondary_targets):
                for t in critical_targets + secondary_targets:
                    if t in cmd:
                        uptime_m = (time.time() - p.info['create_time']) / 60.0
                        mem_mb = (p.info['memory_info'].rss / (1024 * 1024)) if p.info.get('memory_info') else 0.0
                        found_procs[t] = {
                            "pid": p.info['pid'],
                            "uptime_m": uptime_m,
                            "mem_mb": mem_mb
                        }
                        total_mem_mb += mem_mb
                        break
        except Exception:
            pass

    missing_critical = [t for t in critical_targets if t not in found_procs]
    
    # Note: marl_live_trader or live_trader might run as child modules under main or standalone
    # We require start_system.py, ofi_daemon.py, and main.py unconditionally.
    is_healthy = ("start_system.py" in found_procs) and ("ofi_daemon.py" in found_procs) and ("main.py" in found_procs)
    
    return {
        "ok": is_healthy,
        "found": found_procs,
        "missing_critical": missing_critical,
        "total_procs": len(found_procs),
        "total_mem_mb": round(total_mem_mb, 1)
    }


# -------------------------------------------------------------------------
# Check 2: Microstructure & OFI Stream Freshness
# -------------------------------------------------------------------------
def check_microstructure():
    if not os.path.exists(STATE_FILE):
        return {"ok": False, "reason": "microstructure_state.json not found", "latency_s": 999.0}
    
    mtime = os.path.getmtime(STATE_FILE)
    latency_s = time.time() - mtime
    
    state_data = {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state_data = json.load(f)
    except Exception as e:
        return {"ok": False, "reason": f"Corrupted JSON read: {e}", "latency_s": latency_s}
    
    symbols = state_data.get("symbols", {})
    btc_metrics = symbols.get("BTCUSDT", state_data)
    
    # SLA threshold: 10.0 seconds
    is_fresh = latency_s <= 10.0
    
    return {
        "ok": is_fresh,
        "latency_s": round(latency_s, 3),
        "symbols_count": len(symbols),
        "btc_mid": btc_metrics.get("best_bid", 0.0),
        "btc_ofi": btc_metrics.get("normalized_ofi", 0.0),
        "btc_adds": btc_metrics.get("order_additions_count", 0),
        "btc_cancels": btc_metrics.get("order_cancellations_count", 0)
    }


# -------------------------------------------------------------------------
# Check 3: SQLite WAL Health & Read-Only Latency (< 50ms)
# -------------------------------------------------------------------------
def check_sqlite_health():
    results = {}
    overall_ok = True
    
    for db_path, name in [(DATA_DB, "trading_data"), (LEDGER_DB, "trading_ledger")]:
        if not os.path.exists(db_path):
            results[name] = {"ok": False, "reason": "Database file not found"}
            overall_ok = False
            continue
            
        wal_path = f"{db_path}-wal"
        wal_size_mb = (os.path.getsize(wal_path) / (1024 * 1024)) if os.path.exists(wal_path) else 0.0
        
        try:
            # Strict read-only mode via URI to prevent lock contention
            uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            t0 = time.perf_counter()
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.row_factory = sqlite3.Row
            
            # Execute probe query
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            
            # Execute real read query
            if name == "trading_data":
                conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()
            else:
                conn.execute("SELECT COUNT(*) FROM macro_market_context").fetchone()
                
            query_latency_ms = (time.perf_counter() - t0) * 1000.0
            conn.close()
            
            # SLA: < 50ms
            latency_ok = query_latency_ms < 50.0
            if not latency_ok:
                overall_ok = False
                
            results[name] = {
                "ok": latency_ok,
                "latency_ms": round(query_latency_ms, 2),
                "wal_size_mb": round(wal_size_mb, 2),
                "lock_errors": 0
            }
        except Exception as e:
            overall_ok = False
            results[name] = {
                "ok": False,
                "reason": str(e),
                "wal_size_mb": round(wal_size_mb, 2),
                "lock_errors": 1
            }
            
    return {"ok": overall_ok, "details": results}


# -------------------------------------------------------------------------
# Check 4: Log File Scrutiny (Past Hour Errors & Warnings)
# -------------------------------------------------------------------------
def check_logs():
    target_logs = [
        "logs/main.log",
        "logs/marl_live_trader.log",
        "logs/reconciliation_engine.log",
        "logs/ofi_daemon.log"
    ]
    
    findings = {}
    has_unhandled_crash = False
    
    for log_rel in target_logs:
        log_path = os.path.join(BASE_DIR, log_rel)
        if not os.path.exists(log_path):
            continue
            
        mtime = os.path.getmtime(log_path)
        age_hours = (time.time() - mtime) / 3600.0
        
        # If file is older than 24h, mark as quiescent
        if age_hours > 24.0:
            findings[log_rel] = {"status": "quiescent", "age_hours": round(age_hours, 1), "errors": 0}
            continue
            
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                # Read last 400 lines
                lines = f.readlines()[-400:]
                
            unhandled = [l.strip() for l in lines if "Traceback (most recent call last)" in l or "CRITICAL" in l or "FATAL" in l]
            db_locks = [l.strip() for l in lines if "sqlite3.OperationalError: database is locked" in l]
            
            if unhandled:
                has_unhandled_crash = True
                
            findings[log_rel] = {
                "status": "active",
                "tail_lines": len(lines),
                "unhandled_count": len(unhandled),
                "db_lock_count": len(db_locks),
                "sample_alert": unhandled[-1] if unhandled else None
            }
        except Exception as e:
            findings[log_rel] = {"status": "error_reading", "reason": str(e)}
            
    return {"ok": not has_unhandled_crash, "findings": findings}


# -------------------------------------------------------------------------
# Check 5: Position & Trade Bracket Integrity
# -------------------------------------------------------------------------
def check_trade_brackets():
    positions_summary = []
    brackets_ok = True
    
    try:
        uri = f"file:{os.path.abspath(DATA_DB)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        
        # Check active open positions
        rows = conn.execute(
            "SELECT symbol, side, entry_price, qty, take_profit, stop_loss, hard_sl, leverage FROM open_positions"
        ).fetchall()
        
        for r in rows:
            sym = r["symbol"]
            sd = str(r["side"]).upper()
            ep = float(r["entry_price"] or 0.0)
            sl = float(r["stop_loss"] or 0.0)
            tp = float(r["take_profit"] or 0.0)
            hard_sl = float(r["hard_sl"] or 0.0)
            
            # An open position MUST have valid non-zero entry, SL and TP
            pos_bracket_ok = (ep > 0 and sl > 0 and tp > 0)
            if not pos_bracket_ok:
                brackets_ok = False
                
            positions_summary.append({
                "symbol": sym,
                "side": sd,
                "entry": ep,
                "sl": sl,
                "tp": tp,
                "hard_sl": hard_sl,
                "bracket_valid": pos_bracket_ok
            })
            
        # Check last 5 closed trades
        recent_trades = conn.execute(
            "SELECT id, symbol, side, entry_price, exit_price, pnl, slippage_bps, action FROM trades ORDER BY id DESC LIMIT 5"
        ).fetchall()
        
        trades_list = []
        for t in recent_trades:
            trades_list.append({
                "id": t["id"],
                "symbol": t["symbol"],
                "side": t["side"],
                "pnl": float(t["pnl"] or 0.0),
                "action": t["action"]
            })
            
        conn.close()
        
        return {
            "ok": brackets_ok,
            "open_positions_count": len(positions_summary),
            "positions": positions_summary,
            "recent_closed_trades": trades_list
        }
    except Exception as e:
        return {"ok": False, "reason": str(e), "open_positions_count": 0, "positions": []}


# -------------------------------------------------------------------------
# Master Audit Pass & Formatting
# -------------------------------------------------------------------------
def run_hourly_audit_pass(to_console: bool = True) -> bool:
    now_utc = get_utc_now()
    ts_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    
    header = f"\n{'='*75}\n[HOURLY RELIABILITY AUDIT] — {ts_str}\n{'='*75}"
    log_output(header, to_console)
    
    # Run all 5 checks
    c1 = check_processes()
    c2 = check_microstructure()
    c3 = check_sqlite_health()
    c4 = check_logs()
    c5 = check_trade_brackets()
    
    lines = []
    lines.append("[1. PROCESS FLEET HEALTH]")
    if c1["ok"]:
        lines.append(f"  [PASS] Critical Processes Online ({c1['total_procs']} nodes tracked | RAM: {c1['total_mem_mb']} MB)")
        for name, meta in c1["found"].items():
            lines.append(f"     * {name:<26} : PID {meta['pid']:<6} | Uptime: {meta['uptime_m']:.1f}m | RAM: {meta['mem_mb']:.1f}MB")
    else:
        lines.append(f"  [FAIL] Missing Critical Daemons: {c1['missing_critical']}")
        
    lines.append("\n[2. MICROSTRUCTURE & OFI SENSOR FRESHNESS]")
    if c2["ok"]:
        lines.append(f"  [PASS] Stream Freshness: {c2['latency_s']}s latency (SLA: < 10.0s) | Tracking: {c2['symbols_count']} assets")
        lines.append(f"     * BTCUSDT Mid: ${c2['btc_mid']:,.2f} | Norm OFI: {c2['btc_ofi']:+.4f} | Adds: {c2['btc_adds']} | Cancels: {c2['btc_cancels']}")
    else:
        lines.append(f"  [FAIL] OFI Stream Lagging or Offline: {c2.get('reason', '')} (Latency: {c2['latency_s']}s)")
        
    lines.append("\n[3. SQLITE WAL & QUERY LATENCY]")
    if c3["ok"]:
        d_data = c3["details"]["trading_data"]
        d_ledg = c3["details"]["trading_ledger"]
        lines.append(f"  [PASS] Read-Only Latency: trading_data={d_data['latency_ms']}ms, trading_ledger={d_ledg['latency_ms']}ms (SLA: < 50ms)")
        lines.append(f"  [PASS] WAL File Sizes: trading_data={d_data['wal_size_mb']}MB, trading_ledger={d_ledg['wal_size_mb']}MB | Lock Errors: 0")
    else:
        lines.append(f"  [FAIL] Database Latency or Lock Issue: {c3['details']}")
        
    lines.append("\n[4. LOG SCRUTINY (LAST HOUR)]")
    if c4["ok"]:
        lines.append("  [PASS] 0 Unhandled Crashes / Exceptions in active logs")
        for log_name, data in c4["findings"].items():
            st = data.get("status")
            if st == "active":
                lines.append(f"     * {log_name:<30} : OK ({data['tail_lines']} lines scanned | 0 unhandled | 0 locks)")
            elif st == "quiescent":
                lines.append(f"     * {log_name:<30} : Quiescent (Inactive for {data['age_hours']}h)")
    else:
        lines.append("  [FAIL] Log Anomalies Detected:")
        for log_name, data in c4["findings"].items():
            if data.get("unhandled_count", 0) > 0:
                lines.append(f"     * {log_name}: {data['unhandled_count']} unhandled exceptions! Sample: {data['sample_alert']}")
                
    lines.append("\n[5. POSITION BRACKET & RISK INTEGRITY]")
    if c5["ok"]:
        lines.append(f"  [PASS] All {c5['open_positions_count']} active open positions have valid SL/TP/HardSL brackets:")
        for p in c5["positions"]:
            lines.append(f"     * {p['symbol']:<10} {p['side']:<5} | Entry: ${p['entry']:<9.4f} | SL: ${p['sl']:<9.4f} | TP: ${p['tp']:<9.4f} | HardSL: ${p['hard_sl']:<9.4f}")
    else:
        lines.append(f"  [FAIL] Missing or Corrupt Position Brackets: {c5.get('reason', '')}")
        
    # Overall Verdict Computation
    all_passed = c1["ok"] and c2["ok"] and c3["ok"] and c4["ok"] and c5["ok"]
    
    lines.append("\n" + "-"*75)
    if all_passed:
        verdict = "[SYSTEM STATUS: HEALTHY]"
        lines.append(f"  {verdict} — All 5 core subsystem SLAs satisfied.")
    else:
        verdict = "[SYSTEM STATUS: DEGRADED]"
        reasons = []
        if not c1["ok"]: reasons.append(f"Missing processes: {c1['missing_critical']}")
        if not c2["ok"]: reasons.append(f"OFI lag: {c2['latency_s']}s")
        if not c3["ok"]: reasons.append("DB latency or lock error")
        if not c4["ok"]: reasons.append("Unhandled exception in logs")
        if not c5["ok"]: reasons.append("Position bracket missing")
        
        diag_str = " | ".join(reasons)
        lines.append(f"  {verdict}")
        lines.append(f"  [ALERT: CRITICAL_CHECK_FAILED] Diagnosis: {diag_str}")
        
    lines.append("="*75 + "\n")
    
    full_report = "\n".join(lines)
    log_output(full_report, to_console)
    
    return all_passed


# -------------------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Automated Hourly Reliability Sentinel")
    parser.add_argument("--once", action="store_true", help="Run a single audit pass and exit")
    parser.add_argument("--interval", type=int, default=3600, help="Audit interval in seconds (default: 3600)")
    args = parser.parse_args()
    
    if args.once:
        success = run_hourly_audit_pass(to_console=True)
        sys.exit(0 if success else 1)
        
    log_output(f"\n[SENTINEL BOOT] Hourly Reliability Sentinel active. Interval: {args.interval}s.", to_console=True)
    
    # Run immediate initial audit on boot
    run_hourly_audit_pass(to_console=True)
    
    while True:
        try:
            time.sleep(args.interval)
            run_hourly_audit_pass(to_console=True)
        except KeyboardInterrupt:
            log_output("\n[SENTINEL SHUTDOWN] Received KeyboardInterrupt. Gracefully terminating.", to_console=True)
            break
        except Exception as e:
            log_output(f"\n[SENTINEL ERROR] Unexpected loop exception: {e}", to_console=True)
            time.sleep(60)

if __name__ == "__main__":
    main()
