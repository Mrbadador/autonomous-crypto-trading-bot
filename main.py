"""
main.py — Unified Engine Launcher
===================================
Starts the full multi-process trading fleet: Live Oracle, Regime Classifier,
Macro Sentinel, Asset Scout, Sniper (15m), Berserker (5m fleet), Recovery
Agent, and Post-Mortem Analyser as isolated sub-processes. Handles graceful
shutdown on Ctrl+C or SIGTERM, ensuring all child processes are terminated
cleanly before the launcher exits.

Usage:
    python main.py

All engines share two databases:
  - trading_data.db   : positions, balances, system state, UI telemetry
  - trading_ledger.db : market context, oracle signals, sentiment logs

Each bot uses account-namespaced tables (e.g. berserker_open_positions)
to prevent write conflicts between the Sniper and Berserker fleet.
"""

import json
import multiprocessing
import signal
import sys
import traceback
import sqlite3
from email_dispatcher import send_alert_async, send_hourly_report_async, send_hourly_heartbeat_report_async
import logging
import time
import os
import asyncio
import subprocess
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv

# ENV CUSTODIAN: Must be first — loads .env BEFORE config.py is imported.
# config.py also calls load_dotenv(), but being explicit here guarantees
# the env is populated even if import order ever changes.
_BASE_DIR_BOOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR_BOOT, ".env"))
import dns_fix  # noqa: F401

import config
from warden_60m import Warden60Minute
from sentinel_agent import SentimentSentinel

# V20.1: Instantiate the 60-Minute Macro Warden at launcher level for regime logging
macro_warden_launcher = Warden60Minute(db_path="trading_ledger.db")

# V21.0: Instantiate the Sentiment Sentinel at launcher level for regime logging
news_warden = SentimentSentinel()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "launcher.log")
# [PATCH V32.2-P2] Absolute path for the canonical daemon log,
# consumed by antigravity_rules_handler.py watchdog.
MAIN_LOG_PATH = os.path.join(BASE_DIR, "logs", "main.log")
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH),
        logging.FileHandler(MAIN_LOG_PATH),   # [PATCH V32.2-P2] Watchdog feed
    ],
)
log = logging.getLogger("launcher")

# ENV INTEGRITY CHECK: Report credential status at boot so failures are instantly visible.
_ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
_active_key = getattr(config, "ACTIVE_API_KEY", "") or os.getenv("BITGET_API_KEY", "") or os.getenv("BINANCE_API_KEY", "")
if _active_key:
    log.info(f"[ENV] LAUNCHER ENV OK — {_ex_id.upper()} API_KEY loaded (last 4: ...{_active_key[-4:]}).")
else:
    log.warning(f"[ENV] LAUNCHER: {_ex_id.upper()} API_KEY is empty. Running in paper/sandbox mode.")



def get_today_pnl_and_balances():
    """V16.5: Database helper to query today's closed trade PnL and current balances."""
    db_path = "trading_data.db"
    # Today's date in UTC (matching trade completion timestamps)
    # Using datetime.now(UTC) since timezone-aware objects are standard
    from datetime import UTC
    today_str = datetime.now(UTC).strftime('%Y-%m-%d')
    
    sniper_pnl = 0.0
    berserker_pnl = 0.0
    sniper_bal = config.INITIAL_BALANCE
    berserker_bal = config.INITIAL_BALANCE
    
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        
        # 1. Sum Sniper PnL for today
        cursor = conn.execute("SELECT SUM(pnl) FROM trades WHERE timestamp LIKE ?", (f"{today_str}%",))
        res = cursor.fetchone()
        if res and res[0] is not None:
            sniper_pnl = float(res[0])
            
        # 2. Sum Berserker PnL for today
        cursor = conn.execute("SELECT SUM(pnl) FROM berserker_trades WHERE timestamp LIKE ?", (f"{today_str}%",))
        res = cursor.fetchone()
        if res and res[0] is not None:
            berserker_pnl = float(res[0])
            
        # 3. Retrieve Sniper available balance
        res_s = conn.execute("SELECT value FROM system_state WHERE key = 'balance_sniper'").fetchone()
        if res_s:
            sniper_bal = float(res_s[0])
            
        # 4. Retrieve Berserker available balance
        res_b = conn.execute("SELECT value FROM system_state WHERE key = 'balance_berserker'").fetchone()
        if res_b:
            berserker_bal = float(res_b[0])
            
    except Exception as e:
        log.error(f"[SYSTEM] Error querying today's PnL or balances: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        
    return sniper_bal, sniper_pnl, berserker_bal, berserker_pnl


def get_heartbeat_data():
    """Queries the local DB files and live exchange for the Hourly Heartbeat Protocol."""
    import sqlite3
    
    fleet_pnl = 0.0
    active_exposure_count = 0
    warden_status = "NEUTRAL"
    position_details = []
    
    # 1. In Live mode, calculate Fleet PnL from real session trades or default to 0.0
    if getattr(config, "MODE", "paper") in ["live", "live_pilot"]:
        fleet_pnl = 0.0
        try:
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                for tbl in ["trades", "berserker_trades", "marl_live_trades"]:
                    try:
                        row = conn.execute(f"SELECT SUM(pnl) FROM {tbl} WHERE timestamp >= datetime('now', '-24 hours')").fetchone()
                        if row and row[0] is not None:
                            fleet_pnl += float(row[0])
                    except: pass
        except Exception: pass
    else:
        conn = None
        try:
            conn = sqlite3.connect("trading_ledger.db", timeout=15.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            cursor = conn.execute("SELECT SUM(pnl) FROM trade_lessons")
            row = cursor.fetchone()
            if row and row[0] is not None:
                fleet_pnl = float(row[0])
        except Exception as e:
            log.error(f"[SYSTEM] Error querying Fleet PnL from trading_ledger.db: {e}")
        finally:
            if conn is not None:
                try: conn.close()
                except: pass
        
    # 2. Get Active Exposure (number of unique open nodes/symbols) from trading_data.db
    unique_symbols = set()
    conn = None
    try:
        conn = sqlite3.connect("trading_data.db", timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        
        # Scan positions tables
        for table in ["open_positions", "berserker_open_positions", "marl_live_open_positions"]:
            try:
                cursor_check = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if not cursor_check.fetchone():
                    continue
                
                cursor = conn.execute(f"SELECT symbol, side, entry_price, qty, leverage FROM {table}")
                rows = cursor.fetchall()
                for r in rows:
                    if r[0]:
                        unique_symbols.add(r[0])
                        position_details.append({
                            "symbol": r[0],
                            "side": r[1],
                            "entry_price": r[2],
                            "qty": r[3],
                            "leverage": r[4] if len(r) > 4 and r[4] is not None else 1.0,
                            "source_table": table
                        })
            except Exception as tbl_err:
                log.warning(f"[SYSTEM] Warning querying open positions from {table}: {tbl_err}")
                
        active_exposure_count = len(unique_symbols)
        
        # 3. Retrieve Warden Status (Current Macro Regime)
        try:
            cursor = conn.execute("SELECT regime FROM market_regime ORDER BY rowid DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                warden_status = str(row[0])
        except Exception: pass
            
    except Exception as e:
        log.error(f"[SYSTEM] Error querying active exposure or regime from trading_data.db: {e}")
    finally:
        if conn is not None:
            try: conn.close()
            except: pass
        
    # Fallback for Warden Status: check trading_ledger.db market_context signal_readiness
    if warden_status == "UNKNOWN" or not warden_status:
        conn = None
        try:
            conn = sqlite3.connect("trading_ledger.db", timeout=15.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=15000;")
            cursor = conn.execute("SELECT signal_readiness, trend_strength FROM market_context ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                sig_readiness = float(row[0]) if row[0] is not None else 0.50
                trend_str = float(row[1]) if row[1] is not None else 0.0
                if sig_readiness > 0.70:
                    warden_status = f"TRENDING (Readiness: {sig_readiness:.2f})"
                elif sig_readiness < 0.30:
                    warden_status = f"MEAN_REVERSION (Readiness: {sig_readiness:.2f})"
                else:
                    warden_status = f"NEUTRAL (Chop: {trend_str:+.2f})"
        except Exception as e:
            log.warning(f"[SYSTEM] Fallback Warden status query failed: {e}")
        finally:
            if conn is not None:
                try: conn.close()
                except: pass
            
    # 4. Get balances and PnLs for Section 4 stats — V_AUDIT_FIX: call live DB helper (was hardcoded $58.77 / $0.00)
    try:
        sniper_bal, sniper_pnl, berserker_bal, berserker_pnl = get_today_pnl_and_balances()
    except Exception as _pnl_err:
        log.warning(f"[HEARTBEAT] get_today_pnl_and_balances failed: {_pnl_err}")
        sniper_bal = berserker_bal = float(getattr(config, "INITIAL_BALANCE", 58.77))
        sniper_pnl = berserker_pnl = 0.0

    try:
        import sqlite3 as _sq
        _db_path = os.path.join(BASE_DIR, "trading_data.db")
        with _sq.connect(_db_path, timeout=5.0) as _c:
            _row = _c.execute("SELECT value FROM system_state WHERE key='balance_marl_live'").fetchone()
            # S2-14 FIX: Use standard UTC date boundary for marl_live_trades query
            from datetime import UTC as _UTC
            today_utc = datetime.now(_UTC).strftime('%Y-%m-%d')
            _row2 = _c.execute("SELECT COALESCE(SUM(pnl),0.0) FROM marl_live_trades WHERE timestamp LIKE ?",
                               (f"{today_utc}%",)).fetchone()
            marl_live_pnl = float(_row2[0]) if _row2 else 0.0
    except Exception:
        marl_live_bal = sniper_bal + berserker_bal
        marl_live_pnl = 0.0

    stats_data = {
        "sniper_bal": sniper_bal,
        "sniper_pnl": sniper_pnl,
        "berserker_bal": berserker_bal,
        "berserker_pnl": berserker_pnl,
        "marl_live_bal": marl_live_bal,
        "marl_live_pnl": marl_live_pnl
    }

    return fleet_pnl, active_exposure_count, warden_status, position_details, stats_data


# ---------------------------------------------------------------------------
# Engine entry-points (each runs in its own process / Python interpreter)
# ---------------------------------------------------------------------------
def _run_live_oracle():
    """Entry point for the Live Oracle Ingestion process."""
    import subprocess
    import sys
    log.info("[ORACLE] Spawning live_oracle.py subprocess...")
    try:
        subprocess.run([sys.executable, "live_oracle.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

def _run_regime_classifier():
    """Entry point for the Regime Classifier process."""
    import subprocess
    import sys
    log.info("[REGIME] Spawning regime_classifier.py subprocess...")
    try:
        subprocess.run([sys.executable, "regime_classifier.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

def _run_macro_sentinel():
    """Entry point for the Macro Sentinel process."""
    import subprocess
    import sys
    log.info("[MACRO] Spawning macro_sentinel.py subprocess...")
    try:
        subprocess.run([sys.executable, "macro_sentinel.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

def _run_asset_scout():
    """Entry point for the Asset Scout process."""
    import subprocess
    import sys
    log.info("[SCOUT] Spawning asset_scout.py subprocess...")
    try:
        subprocess.run([sys.executable, "asset_scout.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

def _run_post_mortem():
    """Entry point for the Post-Mortem Analysis process."""
    import subprocess
    import sys
    log.info("[POST-MORTEM] Spawning post_mortem_agent.py subprocess...")
    try:
        subprocess.run([sys.executable, "post_mortem_agent.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

def _run_recovery_agent():
    """Entry point for the Recovery Agent process."""
    import subprocess
    import sys
    log.info("[RECOVERY] Spawning recovery_agent.py subprocess...")
    try:
        subprocess.run([sys.executable, "recovery_agent.py"], env=os.environ)
    except KeyboardInterrupt:
        pass

# ---------------------------------------------------------------------------

def _run_sniper(log_name="sniper"):
    """Entry point for the Sniper (15m) process."""
    import asyncio
    from live_trader import LiveTrader
    log.info(f"[{log_name.upper()}] Engine process started.")
    try:
        asyncio.run(LiveTrader(log_name=log_name).run())
    except KeyboardInterrupt:
        log.info(f"[{log_name.upper()}] Received shutdown signal.")
    except Exception as e:
        log.exception(f"[{log_name.upper()}] Fatal crash: {e}")
    finally:
        log.info(f"[{log_name.upper()}] Engine process exiting.")



class FleetState:
    """
    V23.1 Shared state synchronizer for concurrent fleet operations.
    Arbitrates Alpha Strike priority and hard enforces Portfolio heat limits.
    """
    def __init__(self, db_path="trading_data.db", max_trades=3):
        self.db_path = db_path
        self.MAX_CONCURRENT_TRADES = max_trades
        self.opportunity_map = {} # key: candle_time -> val: {symbol: vol_ratio}
        import asyncio
        self.lock = asyncio.Lock()

    async def register_opportunity(self, candle_time, symbol, vol_ratio):
        async with self.lock:
            if candle_time not in self.opportunity_map:
                self.opportunity_map[candle_time] = {}
            self.opportunity_map[candle_time][symbol] = vol_ratio
            
            # Pruning threshold
            if len(self.opportunity_map) > 8:
                try:
                    oldest = sorted(self.opportunity_map.keys())[0]
                    del self.opportunity_map[oldest]
                except Exception: pass

    async def is_alpha_leader(self, candle_time, symbol):
        import sqlite3
        import json
        import asyncio
        from datetime import datetime
        async with self.lock:
            contenders = self.opportunity_map.get(candle_time, {})
            if not contenders: 
                return False
            
            # The Alpha Strike: highest volume displacement wins
            leader = max(contenders, key=contenders.get)
            vol = contenders[leader]
            
            # Log Alpha status globally for Dashboard retrieval
            try:
                def _write_db():
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA busy_timeout=30000;")
                    with conn:
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)", 
                                    ("fleet_alpha", json.dumps({"symbol": leader, "vol_ratio": float(vol)}), datetime.now(UTC).isoformat()))
                    conn.close()
                await asyncio.to_thread(_write_db)
            except Exception: pass

            return leader == symbol

    async def get_active_positions_count(self):
        """V31.0 P1: Hardened with timeout, WAL, and retry guard.
        Previously had no timeout or retry — could return 0 during a concurrent
        IMMEDIATE transaction, allowing portfolio overflow past MAX_CONCURRENT_TRADES.
        """
        import sqlite3
        import asyncio
        import time as _time
        for attempt in range(5):
            try:
                def _read_db():
                    conn = sqlite3.connect(self.db_path, timeout=10.0)
                    conn.execute("PRAGMA journal_mode=WAL")
                    rows = conn.execute("SELECT qty, entry_price FROM berserker_open_positions").fetchall()
                    conn.close()
                    count = 0
                    for (q, ep) in rows:
                        if (float(q or 0) * float(ep or 0) / config.LEVERAGE) >= 0.05:
                            count += 1
                    return count
                return await asyncio.to_thread(_read_db)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 4:
                    await asyncio.sleep(0.2)
                else:
                    log.error(f"[FLEET] Circuit breaker count failed: {e}. Defaulting to MAX to block trade.")
                    return self.MAX_CONCURRENT_TRADES  # Fail-safe: block the trade
            except Exception as e:
                log.error(f"[FLEET] Circuit breaker count error: {e}. Defaulting to MAX.")
                return self.MAX_CONCURRENT_TRADES
        return self.MAX_CONCURRENT_TRADES



def _run_berserker(log_name="berserker"):
    """Entry point for the Berserker (5m) Fleet Orchestrator."""
    import asyncio
    from berserker_trader import BerserkerTrader
    from warden_60m import Warden60Minute
    from sentinel_agent import SentimentSentinel
    from analyst_agent import AnalystAgent
    from risk_manager import RiskManager

    # V36.6: Bind Berserker strictly to config.SYMBOLS (approved 10 liquid top-tier crypto assets)
    TARGET_FLEET = [s for s in getattr(config, "SYMBOLS", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]) if "/" in s]
    log.info(f"[FLEET] Berserker anchored to approved high-liquidity fleet ({len(TARGET_FLEET)} symbols): {TARGET_FLEET}")

    async def launch_fleet():
        log.info(f"[FLEET] Initializing high-resource shared instances for {len(TARGET_FLEET)} asset nodes.")
        
        # V30.1: Shared high-resource agents to prevent model duplication and event-loop starvation
        from strategist_agent import StrategistAgent
        shared_warden = Warden60Minute(db_path="trading_ledger.db")
        shared_sentinel = SentimentSentinel()
        shared_analyst = AnalystAgent()
        shared_strategist = StrategistAgent()
        shared_risk_manager = RiskManager(historical_win_rate=0.65, avg_risk_reward=1.5, max_risk_cap=0.05)
        
        # V23.1: Intra-Fleet Coordination Cluster
        shared_fleet = FleetState(db_path="trading_data.db", max_trades=3)  # V24.5 P0: correct DB for open positions

        tasks = []
        for idx, symbol in enumerate(TARGET_FLEET):
            # V38.0: Spread AI calls evenly — node 0 fires immediately, each subsequent
            # node waits an extra 5 seconds. With 10 symbols this covers 0–45s of the
            # 300s candle window, ensuring no two nodes hit Groq simultaneously.
            ai_call_offset = idx * 5.0
            log.info(f"[FLEET] Initiating staggered node ignition for {symbol} (AI offset: +{ai_call_offset:.0f}s)...")
            node = BerserkerTrader(
                symbol=symbol, 
                shared_warden=shared_warden, 
                shared_sentinel=shared_sentinel,
                fleet_state=shared_fleet,
                shared_analyst=shared_analyst,
                shared_strategist=shared_strategist,
                shared_risk_manager=shared_risk_manager,
                scan_offset_secs=ai_call_offset,
            )
            # Spawn background coroutine instantly and track it
            tasks.append(asyncio.create_task(node.run()))
            
            # V23.3 Staggered Ignition pause prevents SQL "Database is locked" collisions
            await asyncio.sleep(1.5)


        try:
            log.info("[FLEET] All cluster nodes ignition phases complete. Entering swarm loop.")
            await asyncio.gather(*tasks)
        finally:
            log.info("[FLEET] Closing shared resource connections...")
            try:
                await shared_warden.close()
            except Exception as e:
                log.error(f"[FLEET] Error closing shared warden: {e}")
            try:
                await shared_risk_manager.close()
            except Exception as e:
                log.error(f"[FLEET] Error closing shared risk manager: {e}")

    log.info("[FLEET] Starting cluster orchestrator.")
    try:
        asyncio.run(launch_fleet())
    except KeyboardInterrupt:
        log.info("[FLEET] Received cluster shutdown signal.")
    except Exception as e:
        log.exception(f"[FLEET] Cluster critical crash: {e}")
    finally:
        log.info("[FLEET] Cluster shutdown complete.")


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

def main():
    os.makedirs("logs", exist_ok=True)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--berserker":
            log.info("[LAUNCHER] Running Berserker Fleet directly...")
            _run_berserker("berserker")
            sys.exit(0)
        elif arg == "--sniper":
            log.info("[LAUNCHER] Running Sniper Bot directly...")
            _run_sniper("sniper")
            sys.exit(0)

    log.info("=" * 60)
    log.info("  V14.0 COMMAND CENTER — UNIFIED LAUNCHER (MAINTAINER DAEMON)")
    log.info("  Press Ctrl+C to perform a graceful shutdown.")
    log.info("=" * 60)

    # Spawn list is empty because process management is consolidated under start_system.py
    processes = []

    # ---------------------------------------------------------------------------
    # Graceful shutdown handler
    # ---------------------------------------------------------------------------
    def _shutdown(signum, frame):
        log.warning("[LAUNCHER] Shutdown signal received. Terminating all engines...")
        for p in processes:
            if p.is_alive():
                log.info(f"[LAUNCHER] Terminating {p.name} (PID {p.pid})...")
                p.terminate()

        # Wait up to 10 seconds per process for a clean exit
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                log.warning(f"[LAUNCHER] {p.name} did not exit cleanly — forcing kill.")
                p.kill()
            else:
                log.info(f"[LAUNCHER] {p.name} exited cleanly (exit code {p.exitcode}).")

        log.info("[LAUNCHER] All engines stopped. Session complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ---------------------------------------------------------------------------
    # V13.1 Async Tasks: Health Monitor + Nightly Auditor
    # ---------------------------------------------------------------------------

    async def schedule_nightly_auditor():
        """
        V23.0 OFFLOADED: Moved to Windows Native Task Scheduler for absolute Ryzen stability.
        The 'LlamaTradingBot_NightlySequence' task now manages this workflow at 12:05 AM Daily.
        """
        log.info("[SYSTEM] Core Audit/Train sequence offloaded to Windows Task Scheduler (Absolute Reliability Mode).")
        while True:
            await asyncio.sleep(86400) 

    async def run_alpha_auditor_periodically():
        """V26.1 Forensic Alpha Auditor dispatcher (runs every 24h)"""
        interval = 24 * 3600 # 24 hours
        while True:
            log.info("[SYSTEM] Dispatching scheduled Forensic Alpha Auditor...")
            try:
                def _run():
                    import subprocess
                    return subprocess.run(
                        [sys.executable, "alpha_auditor.py"],
                        capture_output=True,
                        text=True,
                        timeout=600.0
                    )
                res = await asyncio.to_thread(_run)
                if res.returncode == 0:
                    log.info("[SYSTEM] Alpha Auditor run complete.")
                else:
                    log.error(f"[SYSTEM] Alpha Auditor error: {res.stderr}")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to dispatch Alpha Auditor: {e}")
            
            await asyncio.sleep(interval)

    async def run_quant_analysis_periodically():
        """V15.9: Automated Deep Quant Analysis Dispatcher"""
        # 12 hours in seconds
        interval = 12 * 3600
        
        while True:
            log.info("[SYSTEM] Triggering scheduled Deep Quant analysis...")
            try:
                def _run():
                    import subprocess
                    return subprocess.run(
                        [sys.executable, "quant_analyst.py"],
                        capture_output=True,
                        text=True,
                        timeout=1800.0
                    )
                res = await asyncio.to_thread(_run)
                if res.returncode == 0:
                    log.info("[SYSTEM] Deep Quant analysis completed successfully.")
                    await send_alert_async("Agent Success: Quant Analyst", "The 12-hour Deep Quant audit has completed and new trading rules are active.")
                else:
                    log.error(f"[SYSTEM] Deep Quant analysis failed: {res.stderr}")
                    await send_alert_async("Agent Error: Quant Analyst", f"The Quant audit failed: {res.stderr[:200]}")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to dispatch Deep Quant analyst: {e}")
            
            await asyncio.sleep(interval)

    async def run_sentiment_oracle_periodically():
        """V16.0: Sentiment Oracle Dispatcher (News Kill-Switch)"""
        # 15 minutes in seconds
        interval = 15 * 60
        
        while True:
            log.info("[SYSTEM] Triggering Sentiment Oracle (News Audit)...")
            try:
                def _run():
                    import subprocess
                    return subprocess.run(
                        [sys.executable, "sentiment_oracle.py"],
                        capture_output=True,
                        text=True,
                        timeout=180.0
                    )
                res = await asyncio.to_thread(_run)
                if res.returncode == 0:
                    log.info("[SYSTEM] Sentiment Oracle update complete.")
                else:
                    log.error(f"[SYSTEM] Sentiment Oracle failed: {res.stderr}")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to dispatch Sentiment Oracle: {e}")
            
            await asyncio.sleep(interval)

    async def run_whale_watcher_periodically():
        """V16.1: Whale Watcher Dispatcher (Flash-Crash Protection)"""
        # 10 minutes in seconds
        interval = 10 * 60
        
        while True:
            log.info("[SYSTEM] Triggering Whale Watcher (On-Chain Audit)...")
            try:
                def _run():
                    import subprocess
                    ww_path = "whale_watcher.py"
                    if not os.path.exists(ww_path):
                        legacy_path = os.path.join("archive", "legacy_agents", "whale_watcher.py")
                        if os.path.exists(legacy_path):
                            ww_path = legacy_path
                        else:
                            return None
                    return subprocess.run(
                        [sys.executable, ww_path],
                        capture_output=True,
                        text=True,
                        timeout=180.0
                    )
                res = await asyncio.to_thread(_run)
                if res and res.returncode == 0:
                    log.info("[SYSTEM] Whale Watcher update complete.")
                    try:
                        if os.path.exists("whale_state.json"):
                            with open("whale_state.json", "r") as f:
                                w_data = json.load(f)
                                if w_data.get("whale_status") == "DANGER":
                                    await send_alert_async("URGENT: Whale Danger", f"Mega-move detected: {w_data.get('last_mega_move')}")
                    except: pass
                else:
                    err_msg = res.stderr if res else "whale_watcher.py script not found"
                    log.error(f"[SYSTEM] Whale Watcher failed: {err_msg}")
                    await send_alert_async("Agent Error: Whale Watcher", f"Whale audit failed: {err_msg[:200]}")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to dispatch Whale Watcher: {e}")
            
            await asyncio.sleep(interval)

    async def run_volatility_warden_periodically():
        """V16.4: Volatility Warden Dispatcher (Chop Protection)"""
        # 15 minutes in seconds
        interval = 15 * 60
        
        while True:
            log.info("[SYSTEM] Triggering Volatility Warden (Chop Audit)...")
            try:
                def _run():
                    import subprocess
                    return subprocess.run(
                        [sys.executable, "volatility_warden.py"],
                        capture_output=True,
                        text=True,
                        timeout=180.0
                    )
                res = await asyncio.to_thread(_run)
                if res.returncode == 0:
                    log.info("[SYSTEM] Volatility Warden update complete.")
                else:
                    log.error(f"[SYSTEM] Volatility Warden failed: {res.stderr}")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to dispatch Volatility Warden: {e}")
            
            await asyncio.sleep(interval)

    async def run_60m_warden_periodically():
        """V33.0 Fast Macro Trend Warden — evaluates multi-horizon regime and logs it every 5 minutes (300s).
        Persists regime to trading_data.db system_state for real-time Executor fast-path veto.
        """
        interval = 300  # 5 minutes in seconds (accelerated for fast velocity detection)
        
        # First evaluation on startup (after a short warm-up delay)
        await asyncio.sleep(30)
        while True:
            log.info("[SYSTEM] Fast Macro Warden (5m): Evaluating trend regime...")
            try:
                regime = await macro_warden_launcher.determine_regime()
                log.info(f"[60M-WARDEN] Current Macro Regime: {regime}")

                # V32.0: Persist regime to DB cache for Executor fast-path veto
                try:
                    def _write_regime_cache():
                        from datetime import datetime, UTC
                        _db_path = os.path.join(BASE_DIR, "trading_data.db")
                        with sqlite3.connect(_db_path, timeout=10.0) as _c:
                            _c.execute("PRAGMA journal_mode=WAL;")
                            _c.execute("PRAGMA busy_timeout=5000;")
                            _c.execute(
                                "INSERT OR REPLACE INTO system_state (key, value, updated_at) "
                                "VALUES ('warden_60m_regime', ?, ?)",
                                (regime, datetime.now(UTC).isoformat())
                            )
                    await asyncio.to_thread(_write_regime_cache)
                    log.debug(f"[60M-WARDEN] Regime '{regime}' persisted to DB cache.")
                except Exception as _cache_write_err:
                    log.warning(f"[60M-WARDEN] Failed to write regime cache: {_cache_write_err}")

            except Exception as e:
                log.error(f"[SYSTEM] 60-Minute Warden evaluation failed: {e}")
            
            await asyncio.sleep(interval)

    async def run_hourly_report_periodically():
        """V15.0: Sends an Hourly Swarm Heartbeat Report (local DB only, no network calls)"""
        interval = 3600  # 1 hour in seconds
        
        # Warmup delay before first heartbeat briefing
        await asyncio.sleep(30)
        
        while True:
            log.info("[SYSTEM] Triggering Hourly Sovereign Swarm Heartbeat...")
            try:
                # Use to_thread to query database without blocking
                fleet_pnl, active_exposure, warden_status, position_details, stats_data = await asyncio.to_thread(get_heartbeat_data)
                
                # Send the heartbeat report asynchronously (queries local DB exclusively)
                await send_hourly_heartbeat_report_async(
                    fleet_pnl=fleet_pnl,
                    active_exposure=active_exposure,
                    warden_status=warden_status,
                    position_details=position_details,
                    stats_data=stats_data
                )
                log.info("[SYSTEM] Hourly Sovereign Swarm Heartbeat dispatched successfully.")
            except Exception as e:
                log.error(f"[SYSTEM] Failed to generate or send Hourly Heartbeat: {e}")
            
            await asyncio.sleep(interval)

    async def run_wal_watchdog_periodically():
        """
        [V32.0 PERSISTENCE WATCHDOG] Monitors trading_data.db WAL IO health every 5 minutes.
        - Probes round-trip write latency (target: <5ms).
        - Inspects WAL page backlog (target: <800 pages).
        - Fires log.critical and forces TRUNCATE checkpoint if thresholds are breached.
        """
        from db_manager import DatabaseManager
        WAL_IO_SLA_MS = 25.0         # IO wait SLA in milliseconds (calibrated for Windows SQLite WAL)
        WAL_PAGE_WARN_THRESHOLD = 800 # Pages before forcing a checkpoint
        CHECK_INTERVAL_SECS = 120     # 2-minute probe cycle (tightened from 5m — V32.1 boundary calibration)

        db_watcher = DatabaseManager()
        await asyncio.sleep(60)  # Allow system to fully boot before first probe

        global training_active

        while True:
            try:
                # If nightly PPO training / DB compacting is active, we suppress warnings
                if 'training_active' in globals() and training_active:
                    log.info("[WAL-WATCHDOG] Suppressing probe cycle alerts during nightly PPO training/VACUUM window.")
                    await asyncio.sleep(CHECK_INTERVAL_SECS)
                    continue

                latency_ms, page_count = await asyncio.to_thread(
                    lambda: (db_watcher.measure_wal_io_latency(), db_watcher.get_wal_page_count())
                )

                if latency_ms < 0:
                    log.warning("[WAL-WATCHDOG] IO probe failed. DB may be initializing.")
                elif latency_ms > WAL_IO_SLA_MS:
                    log.critical(
                        f"[WAL-WATCHDOG] ⚠ IO WAIT BREACH: {latency_ms:.2f}ms — "
                        f"EXCEEDS SLA of {WAL_IO_SLA_MS:.0f}ms! "
                        f"WAL pages: {page_count}. Forcing TRUNCATE checkpoint..."
                    )
                    await asyncio.to_thread(lambda: db_watcher.checkpoint_wal("TRUNCATE"))
                else:
                    log.info(
                        f"[WAL-WATCHDOG] DB IO healthy. Latency: {latency_ms:.2f}ms ✓ "
                        f"| WAL pages: {page_count}"
                    )

                if page_count > WAL_PAGE_WARN_THRESHOLD:
                    log.warning(
                        f"[WAL-WATCHDOG] WAL page backlog elevated: {page_count} pages. "
                        f"Triggering PASSIVE checkpoint to reduce write amplification."
                    )
                    await asyncio.to_thread(lambda: db_watcher.checkpoint_wal("PASSIVE"))

            except Exception as wal_err:
                log.error(f"[WAL-WATCHDOG] Probe cycle error: {wal_err}")

            await asyncio.sleep(CHECK_INTERVAL_SECS)

    async def run_nightly_trainer_at_midnight():
        """
        [V32.1 NOCTURNAL READINESS] Triggers the offline PPO training cycle at 00:00 UTC.
        Ingests high-fidelity trade data from marl_live_trades (trading_data.db) generated
        by the sovereign local execution pipeline.

        Midnight cycle steps:
          1. Pre-VACUUM  — defragment trading_ledger.db + trading_data.db
          2. PPO training — NightlyTrainer.train_network() on marl_live_trades
          3. Weight sync  — policy_model.pth → strategist_weights_v26.pth (canonical)
          4. Post-VACUUM  — clean fragmentation from batch write operations
        """
        from datetime import datetime, UTC, timedelta
        import shutil

        # V32.1: Declaring shared training status flag to suppress WAL watchdog alerts during training
        global training_active
        training_active = False

        def _vacuum_databases():
            """
            [V32.1 TELEMETRY SANITIZATION] VACUUM both databases to defragment
            free-page lists, compact WAL backlog, and reduce write-latency volatility.
            Enforces a strict 290s timeout (well below 300s lock window) via connection interruption.
            """
            results = {}
            for db_name in ["trading_ledger.db", "trading_data.db"]:
                db_path = os.path.join(BASE_DIR, db_name)
                if not os.path.exists(db_path):
                    results[db_name] = "NOT FOUND"
                    continue
                
                # We track elapsed time to ensure total lock does not exceed 300s SLA.
                start_t = time.time()
                conn = None
                try:
                    size_before = os.path.getsize(db_path)
                    conn = sqlite3.connect(db_path, timeout=10.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA busy_timeout=10000;")
                    
                    # Install a progress handler that aborts the query if it runs for over 290 seconds
                    # sqlite3 progress handler is called periodically during execution. Returning non-zero aborts.
                    def progress_handler():
                        if time.time() - start_t > 290.0:
                            log.error(f"[NIGHTLY-TRAINER] VACUUM on {db_name} exceeded 290s lock window safety limit. Aborting!")
                            return 1 # Non-zero aborts the query
                        return 0
                    
                    conn.set_progress_handler(progress_handler, 100) # Check every 100 instructions
                    
                    # VACUUM rewrites the entire DB into a contiguous file — must NOT be in WAL tx
                    conn.execute("VACUUM;")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.close()
                    conn = None
                    size_after = os.path.getsize(db_path)
                    reclaimed_mb = (size_before - size_after) / 1024 / 1024
                    results[db_name] = f"OK | {size_before/1024/1024:.1f}MB → {size_after/1024/1024:.1f}MB ({reclaimed_mb:+.1f}MB)"
                except Exception as ve:
                    if conn:
                        try:
                            conn.close()
                        except:
                            pass
                    results[db_name] = f"FAILED/ABORTED: {ve}"
            return results

        while True:
            try:
                now_utc = datetime.now(UTC)
                # Calculate seconds until next 00:00 UTC
                next_midnight = (now_utc + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                secs_until_midnight = (next_midnight - now_utc).total_seconds()

                log.info(
                    f"[NIGHTLY-TRAINER] Next PPO optimization window: 00:00 UTC "
                    f"({secs_until_midnight / 3600:.2f}h from now)."
                )
                await asyncio.sleep(secs_until_midnight)

                # 00:00 UTC — Begin sovereign nightly cycle
                log.info("[NIGHTLY-TRAINER] ⏰ 00:00 UTC boundary reached. Initiating Sovereign PPO Optimization...")

                training_active = True
                try:
                    # STEP 1: Pre-training VACUUM — sanitize fragmented DB files
                    log.info("[NIGHTLY-TRAINER] Step 1/4: Pre-training DB VACUUM (telemetry sanitization)...")
                    try:
                        vacuum_results = await asyncio.to_thread(_vacuum_databases)
                        for db_name, result in vacuum_results.items():
                            log.info(f"[NIGHTLY-TRAINER] VACUUM [{db_name}]: {result}")
                    except Exception as vac_err:
                        log.warning(f"[NIGHTLY-TRAINER] Pre-VACUUM failed (non-fatal): {vac_err}")

                    # STEP 2: PPO training cycle
                    log.info("[NIGHTLY-TRAINER] Step 2/4: Running PPO optimization on marl_live_trades...")
                    training_success = False
                    try:
                        def _run_training():
                            from nightly_trainer import NightlyTrainer
                            trainer = NightlyTrainer(
                                memory_file="experience_replay.json",
                                model_path="policy_model.pth"
                            )
                            success = trainer.train_network()
                            if success:
                                trainer.save_model()
                                log.info(
                                    "[NIGHTLY-TRAINER] ✅ PPO cycle complete. "
                                    "Interim weights written: policy_model.pth"
                                )
                            else:
                                log.warning(
                                    "[NIGHTLY-TRAINER] Training aborted — insufficient replay data. "
                                    "Accumulate more closed trades before next cycle."
                                )
                            return success

                        training_success = await asyncio.to_thread(_run_training)
                    except Exception as train_err:
                        log.error(f"[NIGHTLY-TRAINER] PPO training cycle failed: {train_err}")

                    # STEP 3: Sync trained weights to canonical swarm weight file
                    if training_success:
                        log.info("[NIGHTLY-TRAINER] Step 3/4: Syncing weights → strategist_weights_v26.pth...")
                        try:
                             def _sync_weights():
                                 src = os.path.join(BASE_DIR, "policy_model.pth")
                                 dst = os.path.join(BASE_DIR, "strategist_weights_v26.pth")
                                 dst_tmp = dst + ".tmp"
                                 if not os.path.exists(src):
                                     raise FileNotFoundError(f"Source weight file not found: {src}")
                                 shutil.copy2(src, dst_tmp)   # Atomic: write to .tmp first
                                 os.replace(dst_tmp, dst)      # Then rename (atomic on same FS)
                                 size_kb = os.path.getsize(dst) / 1024
                                 log.info(
                                     f"[NIGHTLY-TRAINER] ✅ Canonical weights updated: "
                                     f"strategist_weights_v26.pth ({size_kb:.1f}KB)"
                                 )
                             await asyncio.to_thread(_sync_weights)
                        except Exception as sync_err:
                             log.error(
                                 f"[NIGHTLY-TRAINER] Weight sync failed — "
                                 f"policy_model.pth and strategist_weights_v26.pth are now DIVERGED: {sync_err}"
                             )
                    else:
                        log.info("[NIGHTLY-TRAINER] Step 3/4: Skipped (no training occurred).")

                    # STEP 4: Post-training VACUUM — clean fragmentation from batch writes
                    log.info("[NIGHTLY-TRAINER] Step 4/4: Post-training DB VACUUM...")
                    try:
                        post_vac = await asyncio.to_thread(_vacuum_databases)
                        for db_name, result in post_vac.items():
                            log.info(f"[NIGHTLY-TRAINER] Post-VACUUM [{db_name}]: {result}")
                    except Exception as post_vac_err:
                        log.warning(f"[NIGHTLY-TRAINER] Post-VACUUM failed (non-fatal): {post_vac_err}")

                    log.info("[NIGHTLY-TRAINER] 🏁 Nightly cycle complete. Persistence layer sanitized.")
                finally:
                    training_active = False

                # Guard against rapid re-trigger if clock skew causes sub-1s overshoot
                await asyncio.sleep(60)

            except Exception as e:
                log.error(f"[NIGHTLY-TRAINER] Scheduler error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on unexpected failure

    async def run_watchdog_monitor():
        """V23.0 System Watchdog: triggers emergency alert if trading heartbeat stalls > 10m."""
        check_interval = 60
        threshold_seconds = 10 * 60
        alerted = False
        db_path = os.path.join(BASE_DIR, "trading_data.db")
        
        # Explicitly force the last_heartbeat variable to current time inside initialization
        last_heartbeat = time.time()
        
        log.info("[WATCHDOG] System heartbeat monitoring initialized (Threshold: 10m).")
        
        await asyncio.sleep(120) # Initial warm-up grace period
        
        while True:
            try:
                def get_hb():
                    conn = sqlite3.connect(db_path, timeout=30.0)
                    try:
                        # S2-12 FIX: Check general fleet/berserker heartbeat keys
                        r = conn.execute("""
                            SELECT value FROM system_state 
                            WHERE key LIKE 'heartbeat_%' OR key LIKE 'HEARTBEAT_%' OR key IN ('reconciliation_heartbeat', 'macro_heartbeat')
                            ORDER BY updated_at DESC LIMIT 1
                        """).fetchone()
                        return r[0] if r else None
                    finally:
                        conn.close()

                hb_str = await asyncio.to_thread(get_hb)
                if hb_str:
                    hb_time = datetime.fromisoformat(hb_str)
                    db_time = hb_time.timestamp()
                    # MUST NOT read an old timestamp from the DB upon startup
                    if db_time > last_heartbeat:
                        last_heartbeat = db_time
                
                diff = time.time() - last_heartbeat
                
                if diff > threshold_seconds:
                    if not alerted:
                        log.critical(f"[WATCHDOG ALERT] Berserker 5m loop hasn't pulsed in {diff/60:.1f} minutes!")
                        await send_alert_async(
                            "CRITICAL ERROR: Berserker Heartbeat Failed", 
                            f"The Berserker 5-minute loop has ceased heartbeat operations for {diff/60:.1f} minutes. Immediate manual investigation recommended."
                        )
                        alerted = True
                else:
                    if alerted:
                        log.info("[WATCHDOG RECOVERED] Berserker pulses detected. Clearing alert.")
                        alerted = False
            except Exception as w_err:
                log.error(f"[WATCHDOG] Error checking status: {w_err}")
            
            await asyncio.sleep(check_interval)

    async def health_monitor_loop():
        log.info("[LAUNCHER] Health monitor active. Watching both engines...")
        try:
            while True:
                for p in processes:
                    if not p.is_alive():
                        exit_code = p.exitcode
                        log.error(
                            f"[LAUNCHER] {p.name} died unexpectedly "
                            f"(exit code {exit_code}). Restarting in 5s..."
                        )
                        await asyncio.sleep(5)

                        # Spawn a fresh replacement process
                        args = ()
                        if "Oracle" in p.name:
                            target = _run_live_oracle
                        elif "Classifier" in p.name:
                            target = _run_regime_classifier
                        elif "Sentinel" in p.name:
                            target = _run_macro_sentinel
                        elif "Scout" in p.name:
                            target = _run_asset_scout
                        elif "Post-Mortem" in p.name:
                            target = _run_post_mortem
                        elif "Sniper" in p.name:
                            target, args = _run_sniper, ("sniper",)
                        elif "Recovery" in p.name:
                            # P1-PATCH: Explicit recovery agent routing.
                            # Without this, a crashed Recovery-Agent was resurrected as
                            # a duplicate Berserker fleet via the else fallback.
                            target = _run_recovery_agent
                        else:
                            target, args = _run_berserker, ("berserker",)
                            
                        import multiprocessing as mp
                        ctx = mp.get_context("spawn")
                        new_p = ctx.Process(target=target, args=args, name=p.name, daemon=False)
                        new_p.start()
                        log.info(f"[LAUNCHER] {new_p.name} restarted (PID {new_p.pid})")

                        # Replace in list
                        idx = processes.index(p)
                        processes[idx] = new_p

                await asyncio.sleep(10)  # Check health every 10 seconds
        except asyncio.CancelledError:
            log.info("[LAUNCHER] Health monitor loop stopped.")

    async def portfolio_hud_logger():
        """V24.0 Terminal Portfolio HUD: Synchronizes DB state to terminal every 5m."""
        db_path = os.path.join(BASE_DIR, "trading_data.db")
        
        def get_active_positions():
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            try:
                cursor = conn.execute("SELECT symbol, side, entry_price FROM berserker_open_positions")
                return cursor.fetchall()
            except Exception as e:
                raise e
            finally:
                conn.close()

        while True:
            try:
                rows = await asyncio.to_thread(get_active_positions)
                
                if not rows:
                    log.info("[PORTFOLIO-HUD] Active Positions: 0/3. Scanning for Alpha...")
                else:
                    pos_strings = []
                    for sym, side, price in rows:
                        pos_strings.append(f"{sym} ({side.capitalize()}) Entry: ${price:.2f}")
                    summary = " | ".join(pos_strings)
                    log.info(f"[PORTFOLIO-HUD] Active Positions: {len(rows)}/3 | {summary}")
            except Exception as loop_e:
                log.error(f"[PORTFOLIO-HUD] Database error: {loop_e}")
            
            await asyncio.sleep(300)

    async def main_async():
        # V25.4 Block 4 Armor Initialization
        try:
            db_path = os.path.join(BASE_DIR, "trading_data.db")
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('block_3_status', 'Block 3 Armor is active')")
            conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('system_status', 'V25.0_INSTITUTIONAL_READY')")
            conn.commit()
            conn.close()
            log.info("[SYSTEM] Block 3 Armor (Correlation Guard + OFI Veto) Initialized.")
            log.info("[SYSTEM] Status Updated to V25.0_INSTITUTIONAL_READY.")
        except Exception as e:
            log.error(f"[SYSTEM] Failed to write system status to DB: {e}")

        # Start background tasks
        auditor_task       = asyncio.create_task(schedule_nightly_auditor())
        quant_task         = asyncio.create_task(run_quant_analysis_periodically())
        sentiment_task     = asyncio.create_task(run_sentiment_oracle_periodically())
        whale_task         = asyncio.create_task(run_whale_watcher_periodically())
        vol_task           = asyncio.create_task(run_volatility_warden_periodically())
        report_task        = asyncio.create_task(run_hourly_report_periodically())
        warden_60m_task    = asyncio.create_task(run_60m_warden_periodically())
        watchdog_task      = asyncio.create_task(run_watchdog_monitor())
        hud_task           = asyncio.create_task(portfolio_hud_logger())
        alpha_auditor_task = asyncio.create_task(run_alpha_auditor_periodically())
        # V32.0 Sovereign Infrastructure Tasks
        wal_watchdog_task    = asyncio.create_task(run_wal_watchdog_periodically())
        nightly_trainer_task = asyncio.create_task(run_nightly_trainer_at_midnight())
        
        try:
            # Wait for them (they run forever)
            await asyncio.gather(
                auditor_task, quant_task, sentiment_task,
                whale_task, vol_task, report_task, warden_60m_task,
                watchdog_task, hud_task, alpha_auditor_task,
                wal_watchdog_task, nightly_trainer_task
            )
        finally:
            log.info("[SYSTEM] Closing launcher resources...")
            try:
                await macro_warden_launcher.close()
                log.info("[SYSTEM] Closed macro_warden_launcher.")
            except Exception as e:
                log.error(f"[SYSTEM] Error closing macro_warden_launcher: {e}")

    # Silence ccxt base exchange warning
    logging.getLogger("ccxt").setLevel(logging.ERROR)
    
    # Custom asyncio exception handler to silence unclosed sessions warning
    def custom_handler(loop, context):
        msg = context.get("message", "")
        if "Unclosed client session" in msg or "unclosed" in msg.lower():
            return
        loop.default_exception_handler(context)

    async def main_run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(custom_handler)
        await main_async()

    # Start the async event loop
    try:
        asyncio.run(main_run())
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    # Required on Windows for multiprocessing 'spawn' context
    multiprocessing.freeze_support()
    
    # V15.7: Explicit Boot Diagnostic
    log.info(f"[SYSTEM] Launcher Booting... Balance=${config.INITIAL_BALANCE}, Risk={config.RISK_PER_TRADE_PCT}%")
    
    main()
