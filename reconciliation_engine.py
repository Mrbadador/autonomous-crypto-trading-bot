"""
reconciliation_engine.py — Sovereign Fleet Position & Exchange Reconciler

Key Capabilities:
1. Real-time desync detection between local database open positions and exchange reality.
2. Cross-fleet auditing for Sniper, Berserker, MARL, and Mean-Reversion open positions.
3. Ghost / Orphan position cleanup: detects positions where live price has blown past SL/TP
   and automatically executes paper or live closing logic.
4. Live Bitget Futures reconciliation: Queries Bitget UTA and CCXT to detect
   exchange-side stop-outs or fills and synchronizes local records.
5. Emits telemetry to system_state for real-time monitoring on the Streamlit Command HUD.
"""

import os
import sys
import time
import json
import sqlite3
import logging
import asyncio
from datetime import datetime, UTC
from dotenv import load_dotenv

# Ensure environment is loaded from the script directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
import dns_fix  # noqa: F401
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RECONCILER] %(message)s"
)
log = logging.getLogger("reconciler")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")

RECONCILIATION_INTERVAL_SECS = 30  # Run full audit every 30 seconds

FLEET_TABLES = [
    {"table": "open_positions", "bot": "Sniper", "closed_table": "trades", "is_live": True, "closed_status": "closed", "unreconciled_status": "unreconciled"},
    {"table": "berserker_open_positions", "bot": "Berserker", "closed_table": "berserker_trades", "is_live": True, "closed_status": "closed", "unreconciled_status": "unreconciled"},
    {"table": "marl_live_open_positions", "bot": "MARL", "closed_table": "marl_live_trades", "is_live": True, "closed_status": "closed", "unreconciled_status": "unreconciled"},
    {"table": "swing_open_positions", "bot": "Swing", "closed_table": "swing_trades", "is_live": True, "closed_status": "CLOSED", "unreconciled_status": "UNRECONCILED"},
    {"table": "mean_reversion_open_positions", "bot": "MeanRev", "closed_table": "mean_reversion_trades", "is_live": False, "closed_status": "closed", "unreconciled_status": "unreconciled"},
]

# In-memory deduplication tracker for closure alerts to prevent email spam
_recently_alerted_closures = {}  # symbol -> timestamp
_background_tasks = set()


def update_system_state(key: str, value: str):
    """Safely writes a key-value record to the system_state table."""
    try:
        with sqlite3.connect(DATA_DB, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(
                "INSERT INTO system_state (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, value)
            )
    except Exception as e:
        log.debug(f"Failed to update system_state ({key}): {e}")

def validate_exit_price_plausibility(
    symbol: str,
    side: str,
    entry_p: float,
    exit_p: float,
    size: float = 0.0,
    pnl: float | None = None,
) -> bool:
    """
    Validates that an exit price is economically plausible relative to entry price.
    Prevents corrupt fills from other symbols or failed order books from being recorded
    into swing_trades or trades tables.

    Rules:
    1. Both entry_p and exit_p must be positive numbers.
    2. Exit price must not deviate by more than 30% from entry price (0.70 <= exit_p / entry_p <= 1.40).
    3. If size > 0 and pnl is provided:
       - For LONG: if pnl > 0.5, exit_p cannot be significantly below entry_p (exit_p >= entry_p * 0.95).
       - For SHORT: if pnl > 0.5, exit_p cannot be significantly above entry_p (exit_p <= entry_p * 1.05).
    """
    try:
        e_p = float(entry_p)
        x_p = float(exit_p)
        if e_p <= 0.0 or x_p <= 0.0:
            log.warning(f"[EXIT-PRICE-GUARD] Implausible non-positive price for {symbol}: entry={e_p}, exit={x_p}")
            return False

        ratio = x_p / e_p
        # Allow wide-ranging altcoin swings (up to ±45-50% moves) while catching cross-symbol fill pollution
        # (cross-fill pollution exhibits extreme ratios like <0.50 or >1.70).
        if ratio < 0.50 or ratio > 1.70:
            log.warning(
                f"[EXIT-PRICE-GUARD] Extreme price divergence for {symbol} ({side}): "
                f"entry={e_p}, exit={x_p} (ratio={ratio:.4f} outside [0.50, 1.70])"
            )
            return False

        if pnl is not None and size is not None and float(size) > 0.0:
            pnl_val = float(pnl)
            norm_side = str(side).upper()
            if "LONG" in norm_side:
                if pnl_val > 0.5 and x_p < e_p * 0.95:
                    log.warning(
                        f"[EXIT-PRICE-GUARD] Directional contradiction for LONG {symbol}: "
                        f"PnL=${pnl_val:+.2f} but exit={x_p} < entry={e_p}*0.95"
                    )
                    return False
            elif "SHORT" in norm_side:
                if pnl_val > 0.5 and x_p > e_p * 1.05:
                    log.warning(
                        f"[EXIT-PRICE-GUARD] Directional contradiction for SHORT {symbol}: "
                        f"PnL=${pnl_val:+.2f} but exit={x_p} > entry={e_p}*1.05"
                    )
                    return False

        return True
    except Exception as err:
        log.warning(f"[EXIT-PRICE-GUARD] Error validating exit price for {symbol}: {err}")
        return False


def fetch_live_price(symbol: str) -> float | None:
    """Fetches fast live price from Bitget futures ticker (or Binance fallback) or local cache."""
    sym_clean = str(symbol).upper().replace("/", "").replace("-", "")
    
    # 1. Try local JSON microstructure fast feed if available, fresh (<= 300s), and exactly BTC
    try:
        state_file = os.path.join(BASE_DIR, "microstructure_state.json")
        if os.path.exists(state_file) and sym_clean in ("BTCUSDT", "BTCUSDC"):
            file_age = time.time() - os.path.getmtime(state_file)
            if file_age <= 300.0:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    if "best_bid" in data and float(data["best_bid"]) > 0:
                        return float(data["best_bid"])
            else:
                log.debug(f"[PRICE-ORACLE] microstructure_state.json is stale ({file_age:.1f}s > 300s). Bypassing.")
    except Exception:
        pass

    # 2. Fast HTTP request to Bitget USDT-FUTURES ticker (Primary exchange)
    try:
        import requests
        bitget_url = f"https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol={sym_clean}"
        resp = requests.get(bitget_url, timeout=2)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data and len(data) > 0:
                p = float(data[0].get("lastPr", data[0].get("markPrice", 0.0)))
                if p > 0:
                    return p
    except Exception:
        pass

    # 3. Fallback to Bitget Spot ticker if Futures ticker is unreachable
    try:
        import requests
        url = f"https://api.bitget.com/api/v2/spot/market/tickers?symbol={sym_clean}"
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data and len(data) > 0:
                p = float(data[0].get("lastPr", 0.0) or 0.0)
                if p > 0:
                    return p
    except Exception:
        pass
        
    return None

def audit_paper_positions():
    """
    Audits all 4 local fleet position tables.
    Checks if prices have violated Stop-Loss or Take-Profit thresholds,
    or if corrupted records (negative qty, zero price) exist, and auto-heals them.
    """
    audit_summary = {"audited_at": datetime.now(UTC).isoformat(), "desyncs_healed": 0, "positions_active": 0}
    
    for fleet in FLEET_TABLES:
        table = fleet["table"]
        bot = fleet["bot"]
        closed_table = fleet["closed_table"]
        is_live_fleet = fleet.get("is_live", False) and not getattr(config, "PAPER_TRADE", True)
        
        try:
            with sqlite3.connect(DATA_DB, timeout=10.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                
                for row in rows:
                    row_dict = dict(row)
                    symbol = row_dict.get("symbol")
                    if not symbol:
                        continue
                        
                    audit_summary["positions_active"] += 1
                    side = str(row_dict.get("side", "")).lower()
                    entry_p = float(row_dict.get("entry_price", 0.0))
                    qty = float(row_dict.get("qty", 0.0) or row_dict.get("size", 0.0) or 0.0)
                    sl = float(row_dict.get("stop_loss", 0.0) or 0.0)
                    tp = float(row_dict.get("take_profit", 0.0) or 0.0)
                    
                    if qty <= 0 or entry_p <= 0:
                        log.warning(f"[{bot}] Corrupted position detected for {symbol} (qty={qty}, entry={entry_p}). Removing.")
                        conn.execute(f"DELETE FROM {table} WHERE symbol = ?", (symbol,))
                        audit_summary["desyncs_healed"] += 1
                        continue
                        
                    live_p = fetch_live_price(symbol)
                    if not live_p:
                        continue
                        
                    # Live exchange positions must NOT be purged by the local paper auditor;
                    # their lifecycle is governed by exchange native stops and audited by audit_live_exchange_positions()
                    if is_live_fleet:
                        continue

                    # Check for ghost/orphan blowout where price triggered SL/TP but position remained
                    close_reason = None
                    if side == "long":
                        if sl > 0 and live_p <= sl:
                            close_reason = "RECONCILER_SL_TRIGGER"
                        elif tp > 0 and live_p >= tp:
                            close_reason = "RECONCILER_TP_TRIGGER"
                    elif side == "short":
                        if sl > 0 and live_p >= sl:
                            close_reason = "RECONCILER_SL_TRIGGER"
                        elif tp > 0 and live_p <= tp:
                            close_reason = "RECONCILER_TP_TRIGGER"
                            
                    if close_reason:
                        pnl = (live_p - entry_p) * qty if side == "long" else (entry_p - live_p) * qty
                        log.info(f"[{bot}] Auto-healing orphan {side.upper()} on {symbol}: Price ${live_p:,.2f} crossed threshold. Closing with PnL: ${pnl:+.2f}")
                        
                        # Log to closed trade table
                        ledger_write_ok = False
                        try:
                            cur = conn.execute(
                                f"INSERT INTO {closed_table} (timestamp, symbol, side, entry_price, exit_price, pnl, exit_reason) "
                                f"VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)",
                                (symbol, side, entry_p, live_p, pnl, close_reason)
                            )
                            closed_id = cur.lastrowid
                            # 8.5 Model Calibration Logging
                            try:
                                from model_calibration_tracker import ModelCalibrationTracker
                                tracker = ModelCalibrationTracker()
                                tracker.log_prediction_outcome(
                                    trade_id=closed_id or 0,
                                    symbol=symbol,
                                    signal_source=bot,
                                    confidence=float(row_dict.get("action_confidence") or 70.0),
                                    action=side,
                                    pnl=pnl
                                )
                            except Exception as cal_err:
                                log.debug(f"Calibration log note: {cal_err}")
                            ledger_write_ok = cur.rowcount == 1
                        except Exception as ce:
                            # Table structure fallback
                            ledger_write_ok = False
                            log.warning(f"[{bot}] Closed-table insert failed for {symbol} — retaining open row: {ce}")
                            
                        # Delete from open positions — only when the ledger row is confirmed
                        if ledger_write_ok:
                            conn.execute(f"DELETE FROM {table} WHERE symbol = ?", (symbol,))
                            audit_summary["desyncs_healed"] += 1
                        else:
                            log.warning(f"[{bot}] DEFERRING prune of {symbol} in {table} — closed-trade row not written; retry next pass")

        except Exception as e:
            log.error(f"Error auditing table {table}: {e}")

    return audit_summary

async def audit_live_exchange_positions():
    """
    Live Exchange Position Reconciler (Active when trading real capital or testnet live API).
    Queries Bitget UTA or Binance Futures positions, matches them against local DB records, and syncs balances.
    """
    if config.MODE == "paper" and getattr(config, "PAPER_TRADE", True):
        return {"mode": "paper_active", "live_reconciliation": "bypassed"}

    ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()

    if ex_id == "bitget":
        uta = None
        try:
            from bitget_uta_client import BitgetUTAClient
            uta = BitgetUTAClient()
            bal = await uta.fetch_balance()
            positions = await uta.fetch_positions()

            now_iso = datetime.now(UTC).isoformat()
            with sqlite3.connect(DATA_DB, timeout=10.0) as conn:
                for b_key in ["balance", "balance_sniper", "balance_berserker", "balance_marl_live"]:
                    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                                 (b_key, str(round(bal, 4)), now_iso))

            # Only proceed if positions fetch was clean
            active_exchange_positions = []
            live_symbols = set()
            funding_telemetry = {}
            for p in positions:
                total_qty = float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0)
                if total_qty > 0:
                    active_exchange_positions.append(p)
                    sym_raw = p.get("symbol", "")
                    sym_clean = sym_raw.replace("USDT", "/USDT") if "/" not in sym_raw and sym_raw.endswith("USDT") else sym_raw
                    live_symbols.add(sym_clean)
                    live_symbols.add(sym_raw)

                    # Extract live accrued funding and fee telemetry per active position
                    t_funding = float(p.get("totalFunding", 0.0) or 0.0)
                    pos_fund_info = {
                        "symbol": sym_clean,
                        "raw_symbol": sym_raw,
                        "side": str(p.get("posSide", "long")).lower(),
                        "total_funding": t_funding,
                        "open_fee_total": float(p.get("openFeeTotal", 0.0) or 0.0),
                        "close_fee_total": float(p.get("closeFeeTotal", 0.0) or 0.0),
                        "cur_realised_pnl": float(p.get("curRealisedPnl", 0.0) or 0.0),
                        "unrealised_pnl": float(p.get("unrealisedPnl", 0.0) or 0.0),
                        "mark_price": float(p.get("markPrice", 0.0) or 0.0),
                        "avg_price": float(p.get("avgPrice", 0.0) or 0.0),
                        "updated_at": now_iso
                    }
                    funding_telemetry[sym_raw] = pos_fund_info
                    funding_telemetry[sym_clean] = pos_fund_info
                    funding_telemetry[sym_raw.upper().replace("/", "").replace("-", "")] = pos_fund_info

            recovered_count = 0
            pruned_count = 0

            # 0. ORPHAN ORDER REAPER: Scan Bitget for abandoned resting limit orders
            # Only reap orders that are not plan/trigger orders and older than 300s (protecting maker grids and TP/SL)
            try:
                unfilled_res = await uta.exchange.private_uta_get_v3_trade_unfilled_orders({"category": "USDT-FUTURES"})
                unfilled_data = unfilled_res.get("data", {})
                unfilled_list = unfilled_data.get("list", []) if isinstance(unfilled_data, dict) else (unfilled_data if isinstance(unfilled_data, list) else [])
                now_ms = time.time() * 1000
                for u_ord in unfilled_list:
                    u_id = u_ord.get("orderId")
                    u_sym = u_ord.get("symbol", "")
                    u_type = str(u_ord.get("orderType", "")).lower()
                    reduce_only = str(u_ord.get("reduceOnly", "")).lower() in ("true", "1")
                    # Do not cancel plan, conditional, or reduce-only (TP/SL) orders!
                    if "plan" in u_type or "trigger" in u_type or reduce_only:
                        continue

                    c_time = float(u_ord.get("createdTime", 0) or 0)
                    age_sec = (now_ms - c_time) / 1000.0 if c_time > 0 else 999.0
                    
                    # Only cancel non-reduce-only limit orders resting beyond 300 seconds
                    if age_sec > 300.0 and u_type == "limit":
                        log.warning(f"[RECONCILER-ORPHAN-REAPER] Found orphaned resting entry limit {u_id} on {u_sym} (age={age_sec:.1f}s). Cancelling from exchange!")
                        try:
                            await uta.cancel_order(u_sym, u_id)
                            log.info(f"[RECONCILER-ORPHAN-REAPER] Successfully cancelled orphaned order {u_id} on {u_sym}")
                        except Exception as c_err:
                            log.error(f"[RECONCILER-ORPHAN-REAPER] Failed to cancel orphaned order {u_id}: {c_err}")
            except Exception as uo_err:
                log.debug(f"[RECONCILER] Unfilled orders check note: {uo_err}")

            with sqlite3.connect(DATA_DB, timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=10000;")
                # Store live position funding telemetry in system_state for Streamlit HUD & risk agents
                try:
                    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('live_position_funding', ?, ?)",
                                 (json.dumps(funding_telemetry), now_iso))
                except Exception as f_err:
                    log.debug(f"[RECONCILER] Note saving live funding telemetry: {f_err}")

                # P0-3 SAFEGUARD: Gather all local open position symbols to check for false-positive purge
                local_open_symbols = set()
                for fleet in FLEET_TABLES:
                    if not fleet.get("is_live", True):
                        continue
                    tbl = fleet["table"]
                    try:
                        t_rows = conn.execute(f"SELECT symbol FROM {tbl}").fetchall()
                        for (ts,) in t_rows:
                            if ts:
                                local_open_symbols.add(ts)
                    except Exception:
                        pass

                # If exchange returned 0 positions globally but local tables have open positions:
                # Verify with targeted single-symbol checks before allowing any deletions!
                exchange_verified_empty = True
                if len(local_open_symbols) > 0 and len(active_exchange_positions) == 0:
                    log.warning(f"[RECONCILER-SAFEGUARD] Global fetch returned 0 positions, but {len(local_open_symbols)} open positions exist locally ({local_open_symbols}). Running targeted verification...")
                    for sym_to_check in list(local_open_symbols):
                        try:
                            target_res = await uta.fetch_positions(symbol=sym_to_check)
                            matching = [p for p in (target_res or []) if float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0) > 0]
                            if matching:
                                exchange_verified_empty = False
                                log.critical(f"[RECONCILER-SAFEGUARD] 🚨 API GLITCH PREVENTED! Global fetch was empty, but targeted query found active position for {sym_to_check}! Aborting false purge.")
                                for mp in matching:
                                    active_exchange_positions.append(mp)
                                    s_r = mp.get("symbol", "")
                                    s_c = s_r.replace("USDT", "/USDT") if "/" not in s_r and s_r.endswith("USDT") else s_r
                                    live_symbols.add(s_c)
                                    live_symbols.add(s_r)
                        except Exception as check_err:
                            exchange_verified_empty = False
                            log.warning(f"[RECONCILER-SAFEGUARD] Targeted verification for {sym_to_check} threw error ({check_err}). Aborting deletions this cycle to protect data.")
                            break

                # 1. Clean up stale local positions that no longer exist on Bitget exchange
                live_exchange_keys = set()
                for p in active_exchange_positions:
                    s_r = p.get("symbol", "")
                    s_c = s_r.replace("USDT", "/USDT") if "/" not in s_r and s_r.endswith("USDT") else s_r
                    sd = str(p.get("posSide", "long")).lower()
                    live_exchange_keys.add((s_c, sd))
                    live_exchange_keys.add((s_r, sd))

                if not exchange_verified_empty and len(active_exchange_positions) == 0:
                    log.warning("[RECONCILER-SAFEGUARD] Skipping position deletion: Exchange state not verified clean.")
                    fleets_to_prune = []
                else:
                    fleets_to_prune = FLEET_TABLES

                for fleet in fleets_to_prune:
                    if not fleet.get("is_live", True):
                        continue
                    tbl = fleet["table"]
                    closed_tbl = fleet.get("closed_table", "trades")
                    bot_name = fleet.get("bot", "Fleet Agent")
                    try:
                        cursor = conn.execute(f"SELECT * FROM {tbl}")
                        col_names = [d[0] for d in cursor.description]
                        rows = cursor.fetchall()
                        for row in rows:
                            row_dict = dict(zip(col_names, row))
                            s_sym = row_dict.get("symbol", "")
                            norm_side = str(row_dict.get("side", "long")).lower()

                            # Grace period check: if position was created < 60s ago, give it time to settle
                            created_ts = row_dict.get("opened_at") or row_dict.get("entry_time") or row_dict.get("timestamp")
                            if created_ts:
                                try:
                                    c_clean = str(created_ts).replace("Z", "+00:00")
                                    c_dt = datetime.fromisoformat(c_clean)
                                    now_dt = datetime.now(UTC) if c_dt.tzinfo else datetime.utcnow()
                                    if (now_dt - c_dt).total_seconds() < 60.0:
                                        continue
                                except Exception:
                                    pass

                            if (s_sym, norm_side) not in live_exchange_keys and (s_sym.replace("/", ""), norm_side) not in live_exchange_keys:
                                # P0 SAFEGUARD: Double-check with a targeted single-symbol fetch to prevent false closure on partial batch drops
                                try:
                                    targeted_chk = await uta.fetch_positions(symbol=s_sym)
                                    matching_live = [
                                        p for p in (targeted_chk or [])
                                        if float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0) > 0
                                        and str(p.get("posSide", "long")).lower() == norm_side
                                    ]
                                    if matching_live:
                                        log.warning(f"[RECONCILER-SHIELD] 🛡️ Partial API drop prevented! Global list missed {s_sym} {norm_side.upper()}, but targeted query confirmed active position. Preserving local record.")
                                        continue
                                except Exception as target_err:
                                    log.warning(f"[RECONCILER-SHIELD] Targeted verification for {s_sym} returned error ({target_err}). Skipping pruning this cycle to prevent accidental deletion.")
                                    continue

                                log.info(f"[RECONCILER] Detected on-exchange closure for {tbl} -> {s_sym} {norm_side.upper()}. Reconciling & notifying...")
                                
                                entry_p = float(row_dict.get("entry_price", 0.0) or 0.0)
                                qty = float(row_dict.get("qty", 0.0) or row_dict.get("size", 0.0) or 0.0)
                                sl = float(row_dict.get("stop_loss", 0.0) or 0.0)
                                tp = float(row_dict.get("take_profit", 0.0) or 0.0)
                                lev = float(row_dict.get("leverage", 5.0) or 5.0)
                                is_long = "long" in norm_side

                                # Ground Truth Fill Query from Bitget UTA V3 fills
                                exit_p = None
                                actual_fee = None
                                actual_pnl = None
                                try:
                                    recent_fills = await uta.get_recent_fills(symbol=s_sym, lookback_mins=30)
                                    # Filter for matching close fills
                                    target_close_side = "close_long" if is_long else "close_short"
                                    matching_fills = [f for f in recent_fills if f.get("tradeSide") == target_close_side]
                                    if matching_fills:
                                        tot_qty = sum(float(f.get("execQty", 0.0) or 0.0) for f in matching_fills)
                                        tot_val = sum(float(f.get("execValue", 0.0) or (float(f.get("execQty", 0.0) or 0.0) * float(f.get("execPrice", 0.0) or 0.0))) for f in matching_fills)
                                        tot_fee = sum(sum(float(fd.get("fee", 0.0) or 0.0) for fd in f.get("feeDetail", [])) if f.get("feeDetail") else float(f.get("fee", 0.0) or 0.0) for f in matching_fills)
                                        fill_pnls = [float(f.get("execPnl", 0.0) or 0.0) for f in matching_fills if f.get("execPnl") is not None]

                                        if tot_qty > 0 and tot_val > 0:
                                            exit_p = tot_val / tot_qty
                                        if tot_fee > 0:
                                            actual_fee = tot_fee
                                        if fill_pnls:
                                            # Entry taker fee (0.06%) + actual close fee
                                            entry_fee_est = entry_p * qty * 0.0006
                                            actual_pnl = sum(fill_pnls) - (entry_fee_est + tot_fee)
                                        log.info(f"[RECONCILER-FILLS] Ground-truth fills for {s_sym}: Avg Price=${exit_p:,.4f}, Total Fees=${tot_fee:.4f}")
                                except Exception as fill_fetch_err:
                                    log.debug(f"[RECONCILER-FILLS-NOTE] Error querying fills for {s_sym}: {fill_fetch_err}")

                                # Handle ground-truth fill availability honestly (Stage 1 Quantitative Invariant)
                                if exit_p is not None and not validate_exit_price_plausibility(s_sym, norm_side, entry_p, exit_p, qty, actual_pnl):
                                    log.error(f"[RECONCILER-CORRUPT-EXIT] Exit price ${exit_p} for {s_sym} failed plausibility check vs entry ${entry_p}. Setting exit_p=None (UNRECONCILED).")
                                    exit_p = None

                                is_unreconciled = False
                                if exit_p is None or exit_p <= 0:
                                    is_unreconciled = True
                                    exit_p = None
                                    gross_pnl = None
                                    est_fee = None
                                    net_pnl = None
                                    roe_val = None
                                    close_type = "UNRECONCILED_CLOSE"
                                    notes_str = f"UNRECONCILED: Position closed on Bitget but fill history unavailable. No fabricated prices."
                                    log.warning(f"[RECONCILER-UNRECONCILED] No ground-truth fills found for {s_sym}. Recording status='UNRECONCILED' with NULL exit_price/pnl.")
                                else:
                                    gross_pnl = (exit_p - entry_p) * qty if is_long else (entry_p - exit_p) * qty
                                    est_fee = actual_fee if actual_fee is not None else ((entry_p * qty * 0.0006) + (exit_p * qty * 0.0006))
                                    net_pnl = actual_pnl if actual_pnl is not None else (gross_pnl - est_fee)
                                    roe_val = ((exit_p - entry_p)/entry_p * lev * 100.0) if is_long and entry_p > 0 else (((entry_p - exit_p)/entry_p * lev * 100.0) if entry_p > 0 else 0.0)

                                # Calculate Real Holding Time
                                holding_hours = 1.0
                                created_ts = row_dict.get("opened_at") or row_dict.get("entry_time") or row_dict.get("timestamp")
                                if created_ts:
                                    try:
                                        c_clean = str(created_ts).replace("Z", "+00:00")
                                        if "T" not in c_clean and " " in c_clean:
                                            c_clean = c_clean.replace(" ", "T")
                                        if "+" not in c_clean and not c_clean.endswith("Z"):
                                            c_clean += "+00:00"
                                        c_dt = datetime.fromisoformat(c_clean)
                                        now_dt = datetime.now(UTC) if c_dt.tzinfo else datetime.utcnow()
                                        holding_hours = max(0.01, (now_dt - c_dt).total_seconds() / 3600.0)
                                    except Exception:
                                        pass

                                if is_unreconciled:
                                    # Attribution already assigned as UNRECONCILED_CLOSE
                                    pass
                                else:
                                    is_tp_price = (exit_p is not None and tp > 0 and ((is_long and exit_p >= tp * 0.998) or (not is_long and exit_p <= tp * 1.002)))
                                    is_breakeven_price = (exit_p is not None and sl > 0 and ((is_long and abs(exit_p - entry_p * 1.0018) / entry_p < 0.003) or (not is_long and abs(exit_p - entry_p * 0.9982) / entry_p < 0.003)))

                                    if is_tp_price:
                                        close_type = "tp_hit"
                                    elif is_breakeven_price or (net_pnl is not None and abs(net_pnl) < (entry_p * qty * 0.003)):
                                        close_type = "breakeven_hit"
                                    else:
                                        close_type = "sl_hit"

                                    pnl_str = f"${net_pnl:+.2f}" if net_pnl is not None else "N/A"
                                    fee_str = f"-${est_fee:.3f}" if est_fee is not None else "N/A"
                                    notes_str = f"On-exchange {close_type.upper()} on Bitget UTA (Net: {pnl_str}, Fee: {fee_str})"
                                
                                # Check if engine already closed & recorded this trade in closed_tbl recently (last 15 minutes)
                                recent_close = conn.execute(
                                    f"SELECT timestamp FROM {closed_tbl} WHERE (symbol = ? OR symbol = ?) AND ABS(entry_price - ?) <= MAX(? * 0.002, 0.0001) AND replace(substr(timestamp, 1, 19), 'T', ' ') >= datetime('now', '-15 minutes') ORDER BY rowid DESC LIMIT 1",
                                    (s_sym, s_sym.replace('/', ''), entry_p, entry_p)
                                ).fetchone()
                                
                                ledger_write_ok = True
                                if not recent_close:
                                    try:
                                        intended_entry_val = row_dict.get("intended_entry_price") or entry_p
                                        actual_fill_entry_val = row_dict.get("actual_fill_price") or entry_p
                                        slippage_bps_val = row_dict.get("slippage_bps") or 0.0
                                        gates_snap_val = row_dict.get("gates_snapshot")
                                        rec_status = fleet.get("unreconciled_status", "unreconciled") if is_unreconciled else fleet.get("closed_status", "closed")

                                        if closed_tbl == "swing_trades":
                                            cur_ins = conn.execute(
                                                """INSERT INTO swing_trades (
                                                    timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct,
                                                    holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price,
                                                    slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd,
                                                    funding_source, gates_snapshot, status
                                                )
                                                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                (
                                                    s_sym, norm_side, entry_p, exit_p, qty, net_pnl, roe_val,
                                                    holding_hours, close_type, notes_str, intended_entry_val, actual_fill_entry_val,
                                                    slippage_bps_val, est_fee, 0.0, est_fee, 0.0,
                                                    "reconciler", gates_snap_val, rec_status
                                                )
                                            )
                                        else:
                                            cur_ins = conn.execute(
                                                f"INSERT INTO {closed_tbl} (timestamp, symbol, side, type, entry_price, exit_price, qty, pnl, balance, notes, status, action) "
                                                "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                                (s_sym, norm_side, close_type, entry_p, exit_p, qty, net_pnl, bal, notes_str, rec_status, norm_side)
                                            )
                                        ledger_write_ok = cur_ins.rowcount == 1
                                    except Exception as ins_e:
                                        ledger_write_ok = False
                                        log.error(f"[RECONCILER] Ledger INSERT FAILED into {closed_tbl} for {s_sym} — retaining open-position row for retry: {ins_e}")

                                # Delete from open positions table ONLY when the ledger row is confirmed
                                # (already recorded by the engine, or our insert reported exactly 1 row).
                                if ledger_write_ok:
                                    conn.execute(f"DELETE FROM {tbl} WHERE symbol = ? AND side = ?", (s_sym, row_dict.get("side", norm_side)))
                                    conn.execute("""
                                        DELETE FROM asset_locks
                                        WHERE symbol = ?
                                          AND locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect')
                                          AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0
                                    """, (s_sym,))
                                    pruned_count += 1
                                else:
                                    log.error(f"[RECONCILER] DEFERRING prune of {tbl}/{s_sym} — closed-trade row not written; will retry next pass")
                                
                                # Anti-Knife Protection: Add symbol to swing_cooldowns if on-exchange SL fired or trade lost
                                if close_type == "sl_hit" or (net_pnl is not None and net_pnl < 0):
                                    try:
                                        from datetime import timedelta, timezone
                                        cd_hours = 6.0
                                        cd_until = (datetime.now(timezone.utc) + timedelta(hours=cd_hours)).isoformat()
                                        cd_now = datetime.now(timezone.utc).isoformat()
                                        conn.execute("""
                                            INSERT OR REPLACE INTO swing_cooldowns (symbol, cooldown_until, reason, created_at)
                                            VALUES (?, ?, ?, ?)
                                        """, (s_sym, cd_until, "ON_EXCHANGE_SL_HIT", cd_now))
                                        log.info(f"[RECONCILER-COOLDOWN] 🛡️ {s_sym} locked in {cd_hours:.0f}h cooldown after on-exchange SL hit ({close_type}).")
                                    except Exception as cd_err:
                                        log.debug(f"[RECONCILER-COOLDOWN-NOTE] {cd_err}")
                                
                                # Dispatch Email Alert (Deduplicated with 15-minute cooldown and engine close check)
                                import time as _tm
                                now_epoch = _tm.time()
                                last_alert_time = _recently_alerted_closures.get(s_sym, 0.0)
                                if recent_close:
                                    log.info(f"[RECONCILER] Position {s_sym} was already closed & settled by engine. Skipping duplicate reconciler alert.")
                                elif now_epoch - last_alert_time < 900.0:
                                    log.info(f"[RECONCILER] Suppressing duplicate closure email for {s_sym} (previously alerted {now_epoch - last_alert_time:.1f}s ago).")
                                else:
                                    _recently_alerted_closures[s_sym] = now_epoch
                                    try:
                                        # BUG-09 FIX: Use async email dispatch to avoid blocking the reconciler event loop
                                        from email_dispatcher import send_trade_log_async
                                        close_price_str = f" at ${exit_p:,.4f}" if exit_p is not None else " (unverified exit price)"
                                        pnl_display = f"${net_pnl:+.4f}" if net_pnl is not None else "N/A"
                                        t_close = asyncio.create_task(send_trade_log_async(
                                            symbol=s_sym,
                                            side=norm_side.upper(),
                                            price=exit_p,
                                            pnl=net_pnl,
                                            status="CLOSED",
                                            trade_context={
                                                "qty": qty,
                                                "pnl": net_pnl,
                                                "leverage": lev,
                                                "entry_price": entry_p,
                                                "engine": f"{bot_name} (Exchange Auto-Close)",
                                                "close_trigger": "BITGET ON-EXCHANGE MATCH",
                                                "close_reason": f"Position was executed and settled directly on Bitget USDT-M matching engine ({close_type.upper()}){close_price_str}.",
                                                "notes": f"Executed directly on Bitget Matching Engine ({close_type.upper()})"
                                            }
                                        ))
                                        _background_tasks.add(t_close)
                                        t_close.add_done_callback(_background_tasks.discard)
                                        log.info(f"[RECONCILER] Dispatched trade closure email for {s_sym} (PnL: {pnl_display})")
                                    except Exception as mail_e:
                                        log.warning(f"[RECONCILER] Failed to send email for {s_sym}: {mail_e}")
                    except Exception as fleet_e:
                        log.debug(f"[RECONCILER] Error processing fleet table {tbl}: {fleet_e}")

                # 2. Check remaining local positions & purge duplicate shadows
                existing_local_keys = set()
                swing_symbols = set()
                try:
                    s_rows = conn.execute("SELECT symbol FROM swing_open_positions").fetchall()
                    for (sr,) in s_rows:
                        swing_symbols.add(sr)
                        swing_symbols.add(sr.replace("/", ""))
                except Exception:
                    pass

                # If an asset is actively managed by swing_open_positions, purge duplicate shadow in open_positions
                for ss in swing_symbols:
                    try:
                        clean_s = ss.replace("USDT", "/USDT") if "/" not in ss and ss.endswith("USDT") else ss
                        raw_s = ss.replace("/", "")
                        conn.execute("DELETE FROM open_positions WHERE symbol = ? OR symbol = ?", (clean_s, raw_s))
                    except Exception:
                        pass

                for fleet in FLEET_TABLES:
                    if not fleet.get("is_live", True):
                        continue
                    tbl = fleet["table"]
                    try:
                        rows = conn.execute(f"SELECT symbol, side FROM {tbl}").fetchall()
                        for (s_sym, s_side) in rows:
                            norm_side = str(s_side or "long").lower()
                            existing_local_keys.add((s_sym, norm_side))
                            existing_local_keys.add((s_sym.replace("/", ""), norm_side))
                    except Exception:
                        pass

                # 3. TWO-WAY SYNC: Recover any live exchange positions missing from local DB or update size
                for p in active_exchange_positions:
                    sym_raw = p.get("symbol", "")
                    sym_clean = sym_raw.replace("USDT", "/USDT") if "/" not in sym_raw and sym_raw.endswith("USDT") else sym_raw
                    side = str(p.get("posSide", "long")).lower()
                    entry_p = float(p.get("avgPrice", 0.0) or p.get("entryPrice", 0.0) or 0.0)
                    qty = float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0)
                    mark_p = float(p.get("markPrice", 0.0) or entry_p)
                    lev = float(p.get("leverage", 5.0) or 5.0)

                    if (sym_clean, side) not in existing_local_keys and (sym_raw, side) not in existing_local_keys:
                        # CRIT-02 FIX: Query dynamic ATR from market_regime for risk-calibrated exit recovery
                        atr_val = None
                        try:
                            atr_row = conn.execute("SELECT atr FROM market_regime WHERE atr IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
                            if atr_row and atr_row[0] and float(atr_row[0]) > 0:
                                atr_val = float(atr_row[0])
                        except Exception:
                            pass

                        if atr_val and atr_val > 0:
                            sl_dist = atr_val * 2.0
                            tp_dist = atr_val * 3.0
                            if side == "long":
                                sl = min(entry_p - sl_dist, mark_p - sl_dist * 0.5)
                                tp = max(entry_p + tp_dist, mark_p + tp_dist * 0.5)
                                hard_sl = min(entry_p - atr_val * 2.5, mark_p - atr_val * 1.5)
                            else:
                                sl = max(entry_p + sl_dist, mark_p + sl_dist * 0.5)
                                tp = min(entry_p - tp_dist, mark_p - tp_dist * 0.5)
                                hard_sl = max(entry_p + atr_val * 2.5, mark_p + atr_val * 1.5)
                            rec_note = f"Auto-synced from Bitget (ATR={atr_val:.2f})"
                        else:
                            # Conservative 2% SL / 4% TP fallback calibrated relative to current mark price
                            if side == "long":
                                tp = max(entry_p * 1.04, mark_p * 1.04)
                                sl = min(entry_p * 0.98, mark_p * 0.97)
                                hard_sl = min(entry_p * 0.97, mark_p * 0.95)
                            else:
                                tp = min(entry_p * 0.96, mark_p * 0.96)
                                sl = max(entry_p * 1.02, mark_p * 1.03)
                                hard_sl = max(entry_p * 1.03, mark_p * 1.05)
                            rec_note = "Auto-synced from Bitget (Fallback 2% SL)"

                        conn.execute(
                            "INSERT OR REPLACE INTO open_positions (symbol, side, entry_price, qty, take_profit, stop_loss, peak_price, entry_time, is_closing, partial_tp_hit, action_confidence, comments, leverage, hard_sl) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 1.0, ?, ?, ?)",
                            (sym_clean, side, entry_p, qty, tp, sl, max(entry_p, mark_p), now_iso, rec_note, lev, hard_sl)
                        )
                        log.warning(f"[RECONCILER-RECOVERY] Auto-recovered active Bitget position into local DB: {sym_clean} {side.upper()} @ ${entry_p:,.4f} (Qty={qty}, SL=${sl:,.2f}, TP=${tp:,.2f}, HardSL=${hard_sl:,.2f})")
                        
                        # Dispatch Trade Open Email Alert
                        try:
                            from email_dispatcher import send_trade_log_async
                            rec_trade_ctx = {
                                "symbol": sym_clean,
                                "side": side.upper(),
                                "price": entry_p,
                                "qty": qty,
                                "stop_loss": sl,
                                "take_profit": tp,
                                "hard_sl": hard_sl,
                                "leverage": lev,
                                "engine": "Bitget UTA (Live Exchange Onboard)",
                                "ai_reasoning": rec_note,
                                "notes": f"Position detected on Bitget UTA exchange matching engine; registered into local fleet with calibrated stops."
                            }
                            t_open = asyncio.create_task(send_trade_log_async(
                                symbol=sym_clean,
                                side=side.upper(),
                                price=entry_p,
                                status="OPENED",
                                trade_context=rec_trade_ctx
                            ))
                            _background_tasks.add(t_open)
                            t_open.add_done_callback(_background_tasks.discard)
                            log.info(f"[RECONCILER] Dispatched trade OPEN alert email for {sym_clean} {side.upper()}")
                        except Exception as em_err:
                            log.debug(f"[RECONCILER] Open alert email note: {em_err}")

                        recovered_count += 1
                        existing_local_keys.add((sym_clean, side))
                    else:
                        # Position already tracked locally — ensure contract qty & entry in open_positions match exchange
                        if sym_clean not in swing_symbols and sym_raw not in swing_symbols:
                            try:
                                conn.execute(
                                    "UPDATE open_positions SET qty = ?, entry_price = ? WHERE (symbol = ? OR symbol = ?) AND side = ?",
                                    (qty, entry_p, sym_clean, sym_raw, side)
                                )
                            except Exception as sync_upd_err:
                                log.warning(f"[RECONCILER-SYNC-WARN] Failed to update open_positions qty/entry for {sym_clean}: {sync_upd_err}")

                # 3. AUTOMATED PROFIT SWEEPER: Auto-close residual dust runners in profit (margin < $0.05 & gain >= 10%)
                for p in active_exchange_positions:
                    q = float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0)
                    if q <= 0: continue
                    ep = float(p.get("avgPrice", 0.0) or p.get("entryPrice", 0.0) or 0.0)
                    mp = float(p.get("markPrice", 0.0) or ep)
                    sd = str(p.get("posSide", "long")).lower()
                    sym_r = p.get("symbol", "")
                    sym_c = sym_r.replace("USDT", "/USDT") if "/" not in sym_r and sym_r.endswith("USDT") else sym_r
                    
                    gain_pct = ((mp - ep) / ep * 100) if sd == "long" else ((ep - mp) / ep * 100)
                    notional_val = q * mp
                    pos_lev = float(p.get("leverage", 5.0) or 5.0)
                    margin_val = notional_val / max(1.0, pos_lev)
                    
                    if margin_val < 0.05 and gain_pct >= 10.0:
                        try:
                            log.info(f"[PROFIT-SWEEP] Auto-sweeping profitable runner: {sym_c} {sd.upper()} Qty={q} (+{gain_pct:.1f}% gain)")
                            await uta.close_position_order(symbol=sym_r, side=sd, qty=q)
                            live_symbols.discard(sym_r)
                            live_symbols.discard(sym_c)
                        except Exception as swe:
                            log.warning(f"[PROFIT-SWEEP] Failed to auto-sweep {sym_c}: {swe}")

                # 4. Finalize & notify on-exchange closures (when a position was filled/stopped out directly on Bitget)
                for fleet in fleets_to_prune:
                    if not fleet.get("is_live", True):
                        continue
                    tbl = fleet["table"]
                    bot_name = fleet["bot"]
                    closed_tbl = fleet["closed_table"]
                    try:
                        cursor = conn.execute(f"SELECT * FROM {tbl}")
                        col_names = [d[0] for d in cursor.description]
                        rows = cursor.fetchall()
                        for row in rows:
                            row_dict = dict(zip(col_names, row))
                            sym = row_dict.get("symbol", "")
                            side = str(row_dict.get("side", "long")).lower()
                            is_long = "long" in side
                            
                            # 1. Grace period check: if position was created < 60s ago, give it time to settle
                            created_ts = row_dict.get("opened_at") or row_dict.get("entry_time") or row_dict.get("timestamp")
                            if created_ts:
                                try:
                                    c_clean = str(created_ts).replace("Z", "+00:00")
                                    if "T" not in c_clean and " " in c_clean:
                                        c_clean = c_clean.replace(" ", "T")
                                    if "+" not in c_clean and not c_clean.endswith("Z"):
                                        c_clean += "+00:00"
                                    c_dt = datetime.fromisoformat(c_clean)
                                    now_dt = datetime.now(UTC) if c_dt.tzinfo else datetime.utcnow()
                                    if (now_dt - c_dt).total_seconds() < 60.0:
                                        continue
                                except Exception:
                                    pass

                            if sym not in live_symbols and sym.replace("/", "") not in live_symbols:
                                # 2. P0 SAFEGUARD: Double-check with a targeted single-symbol fetch to prevent false closure on partial batch drops
                                try:
                                    targeted_chk = await uta.fetch_positions(symbol=sym)
                                    matching_live = [
                                        p for p in (targeted_chk or [])
                                        if float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0) > 0
                                        and str(p.get("posSide", "long")).lower() == side
                                    ]
                                    if matching_live:
                                        log.warning(f"[RECONCILER-SHIELD-STEP4] 🛡️ Partial API drop prevented in Step 4! Global list missed {sym} {side.upper()}, but targeted query confirmed active position. Preserving local record.")
                                        continue
                                except Exception as target_err:
                                    log.warning(f"[RECONCILER-SHIELD-STEP4] Targeted verification for {sym} returned error ({target_err}). Skipping pruning this cycle to prevent accidental deletion.")
                                    continue

                                log.warning(f"[RECONCILER] Detected on-exchange closure for {sym} in {tbl}. Finalizing trade & dispatching alert.")
                                entry_p = float(row_dict.get("entry_price", 0.0) or 0.0)
                                qty = float(row_dict.get("qty", 0.0) or row_dict.get("size", 0.0) or 0.0)
                                sl = float(row_dict.get("stop_loss", 0.0) or 0.0)
                                tp = float(row_dict.get("take_profit", 0.0) or 0.0)
                                lev = float(row_dict.get("leverage", 5.0) or 5.0)
                                
                                # Ground Truth Fill Query from Bitget UTA V3 fills
                                exit_p = None
                                actual_fee = None
                                actual_pnl = None
                                try:
                                    recent_fills = await uta.get_recent_fills(symbol=sym, lookback_mins=30)
                                    target_close_side = "close_long" if is_long else "close_short"
                                    matching_fills = [
                                        f for f in recent_fills
                                        if f.get("tradeSide") == target_close_side
                                        and uta._normalize_symbol(str(f.get("symbol", ""))) == uta._normalize_symbol(sym)
                                    ]
                                    if matching_fills:
                                        tot_qty = sum(float(f.get("execQty", 0.0) or 0.0) for f in matching_fills)
                                        tot_val = sum(float(f.get("execValue", 0.0) or (float(f.get("execQty", 0.0) or 0.0) * float(f.get("execPrice", 0.0) or 0.0))) for f in matching_fills)
                                        tot_fee = sum(sum(float(fd.get("fee", 0.0) or 0.0) for fd in f.get("feeDetail", [])) if f.get("feeDetail") else float(f.get("fee", 0.0) or 0.0) for f in matching_fills)
                                        fill_pnls = [float(f.get("execPnl", 0.0) or 0.0) for f in matching_fills if f.get("execPnl") is not None]

                                        if tot_qty > 0 and tot_val > 0:
                                            exit_p = tot_val / tot_qty
                                        if tot_fee > 0:
                                            actual_fee = tot_fee
                                        if fill_pnls:
                                            entry_fee_est = entry_p * qty * 0.0006
                                            actual_pnl = sum(fill_pnls) - (entry_fee_est + tot_fee)
                                except Exception as fill_err2:
                                    log.debug(f"[RECONCILER-FILLS-NOTE] {fill_err2}")

                                # Plausibility guard
                                if exit_p is not None and not validate_exit_price_plausibility(sym, side, entry_p, exit_p, qty, actual_pnl):
                                    log.error(f"[RECONCILER-CORRUPT-EXIT-STEP4] Exit price ${exit_p} for {sym} failed plausibility check vs entry ${entry_p}. Setting exit_p=None (UNRECONCILED).")
                                    exit_p = None

                                is_unreconciled = False
                                if exit_p is None or exit_p <= 0:
                                    is_unreconciled = True
                                    exit_p = None
                                    gross_pnl = None
                                    est_fee = None
                                    net_pnl = None
                                    roe_val = None
                                    close_type = "UNRECONCILED_CLOSE"
                                    notes_str = "UNRECONCILED: Position closed on Bitget but fill history unavailable. No fabricated prices."
                                    log.warning(f"[RECONCILER-UNRECONCILED-STEP4] No ground-truth fills found for {sym}. Recording status='UNRECONCILED' with NULL exit_price/pnl.")
                                else:
                                    gross_pnl = (exit_p - entry_p) * qty if is_long else (entry_p - exit_p) * qty
                                    funding_val = float(row_dict.get("total_funding", 0.0) or 0.0)
                                    est_fee = actual_fee if actual_fee is not None else ((entry_p * qty * 0.0006) + (exit_p * qty * 0.0006))
                                    net_pnl = actual_pnl if actual_pnl is not None else (gross_pnl + funding_val - est_fee)
                                    roe_val = (
                                        ((exit_p - entry_p) / entry_p * lev * 100.0) if (is_long and entry_p > 0)
                                        else (((entry_p - exit_p) / entry_p * lev * 100.0) if entry_p > 0 else 0.0)
                                    )

                                # Calculate Real Holding Time
                                holding_hours = 1.0
                                if created_ts:
                                    try:
                                        c_clean = str(created_ts).replace("Z", "+00:00")
                                        if "T" not in c_clean and " " in c_clean:
                                            c_clean = c_clean.replace(" ", "T")
                                        if "+" not in c_clean and not c_clean.endswith("Z"):
                                            c_clean += "+00:00"
                                        c_dt = datetime.fromisoformat(c_clean)
                                        now_dt = datetime.now(UTC) if c_dt.tzinfo else datetime.utcnow()
                                        holding_hours = max(0.01, (now_dt - c_dt).total_seconds() / 3600.0)
                                    except Exception:
                                        pass

                                if not is_unreconciled:
                                    is_tp_price = (tp > 0 and ((is_long and exit_p >= tp * 0.998) or (not is_long and exit_p <= tp * 1.002)))
                                    is_breakeven_price = (sl > 0 and ((is_long and abs(exit_p - entry_p * 1.0018) / entry_p < 0.003) or (not is_long and abs(exit_p - entry_p * 0.9982) / entry_p < 0.003)))

                                    if is_tp_price:
                                        close_type = "tp_hit"
                                    elif is_breakeven_price or (net_pnl is not None and abs(net_pnl) < (entry_p * qty * 0.003)):
                                        close_type = "breakeven_hit"
                                    else:
                                        close_type = "sl_hit"

                                    pnl_str = f"${net_pnl:+.2f}" if net_pnl is not None else "N/A"
                                    fee_str = f"-${est_fee:.3f}" if est_fee is not None else "N/A"
                                    notes_str = f"On-exchange {close_type.upper()} on Bitget UTA (Net: {pnl_str}, Fee: {fee_str})"

                                # Check if engine already closed & recorded this trade in closed_tbl recently (last 15 minutes)
                                recent_close = conn.execute(
                                    f"SELECT timestamp FROM {closed_tbl} WHERE (symbol = ? OR symbol = ?) AND ABS(entry_price - ?) <= MAX(? * 0.002, 0.0001) AND replace(substr(timestamp, 1, 19), 'T', ' ') >= datetime('now', '-15 minutes') ORDER BY rowid DESC LIMIT 1",
                                    (sym, sym.replace('/', ''), entry_p, entry_p)
                                ).fetchone()

                                # 1. Log to closed trades table (only if not already recorded)
                                ledger_write_ok = True
                                if not recent_close:
                                    try:
                                        intended_entry_val = row_dict.get("intended_entry_price") or entry_p
                                        actual_fill_entry_val = row_dict.get("actual_fill_price") or entry_p
                                        slippage_bps_val = row_dict.get("slippage_bps") or 0.0
                                        gates_snap_val = row_dict.get("gates_snapshot")
                                        rec_status = fleet.get("unreconciled_status", "unreconciled") if is_unreconciled else fleet.get("closed_status", "closed")

                                        if closed_tbl == "swing_trades":
                                            cur_ins = conn.execute(
                                                """INSERT INTO swing_trades (
                                                    timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct,
                                                    holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price,
                                                    slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd,
                                                    funding_source, gates_snapshot, status
                                                )
                                                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                (
                                                    sym, side, entry_p, exit_p, qty, net_pnl, roe_val,
                                                    holding_hours, close_type, notes_str, intended_entry_val, actual_fill_entry_val,
                                                    slippage_bps_val, est_fee, 0.0, est_fee, funding_val if not is_unreconciled else 0.0,
                                                    "reconciler", gates_snap_val, rec_status
                                                )
                                            )
                                        else:
                                            cur_ins = conn.execute(
                                                f"INSERT INTO {closed_tbl} (timestamp, symbol, side, type, entry_price, exit_price, qty, pnl, balance, notes, status, action) "
                                                "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                                (sym, side, close_type, entry_p, exit_p, qty, net_pnl, bal, notes_str, rec_status, side)
                                            )
                                        ledger_write_ok = cur_ins.rowcount == 1
                                    except Exception as ins_e:
                                        ledger_write_ok = False
                                        log.error(f"[RECONCILER] Ledger INSERT FAILED into {closed_tbl} for {sym} — retaining open-position row for retry: {ins_e}")

                                # 2. Delete from open positions and commit immediately — only when the ledger row is confirmed
                                if ledger_write_ok:
                                    conn.execute(f"DELETE FROM {tbl} WHERE symbol = ? AND side = ?", (sym, row_dict.get("side", side)))
                                    conn.execute("""
                                        DELETE FROM asset_locks
                                        WHERE symbol = ?
                                          AND locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect')
                                          AND (julianday('now') - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0
                                    """, (sym,))
                                    pruned_count += 1
                                else:
                                    log.error(f"[RECONCILER] DEFERRING prune of {tbl}/{sym} — closed-trade row not written; will retry next pass")
                                if (not is_unreconciled) and (close_type == "sl_hit" or (net_pnl is not None and net_pnl < 0)):
                                    try:
                                        from datetime import timedelta, timezone
                                        cd_hours = 6.0
                                        cd_until = (datetime.now(timezone.utc) + timedelta(hours=cd_hours)).isoformat()
                                        cd_now = datetime.now(timezone.utc).isoformat()
                                        conn.execute("""
                                            INSERT OR REPLACE INTO swing_cooldowns (symbol, cooldown_until, reason, created_at)
                                            VALUES (?, ?, ?, ?)
                                        """, (sym, cd_until, "ON_EXCHANGE_SL_HIT", cd_now))
                                        log.info(f"[RECONCILER-COOLDOWN] 🛡️ {sym} locked in {cd_hours:.0f}h cooldown after on-exchange SL hit ({close_type}).")
                                    except Exception as cd_err:
                                        log.debug(f"[RECONCILER-COOLDOWN-NOTE] {cd_err}")
                                conn.commit()

                                # 3. Dispatch Email Alert (Deduplicated with 15-minute cooldown and engine close check)
                                import time as _tm
                                now_epoch = _tm.time()
                                last_alert_time = _recently_alerted_closures.get(sym, 0.0)
                                if recent_close:
                                    log.info(f"[RECONCILER] Position {sym} was already closed & settled by engine. Skipping duplicate reconciler alert.")
                                elif now_epoch - last_alert_time < 900.0:
                                    log.info(f"[RECONCILER] Suppressing duplicate closure email for {sym} (previously alerted {now_epoch - last_alert_time:.1f}s ago).")
                                else:
                                    _recently_alerted_closures[sym] = now_epoch
                                    try:
                                        from email_dispatcher import send_trade_log_async
                                        t_close = asyncio.create_task(send_trade_log_async(
                                            symbol=sym,
                                            side=side.upper(),
                                            price=exit_p,
                                            pnl=net_pnl,
                                            status="UNRECONCILED" if is_unreconciled else "CLOSED",
                                            trade_context={
                                                "qty": qty,
                                                "pnl": net_pnl,
                                                "leverage": lev,
                                                "entry_price": entry_p,
                                                "engine": f"{bot_name} (Exchange Auto-Close)",
                                                "close_trigger": "BITGET ON-EXCHANGE MATCH",
                                                "close_reason": f"Position was executed and settled directly on Bitget USDT-M matching engine ({close_type.upper()}).",
                                                "notes": notes_str
                                            }
                                        ))
                                        _background_tasks.add(t_close)
                                        t_close.add_done_callback(_background_tasks.discard)
                                        log.info(f"[RECONCILER] Dispatched trade closure email task for {sym} (PnL: {net_pnl})")
                                    except Exception as mail_e:
                                        log.warning(f"[RECONCILER] Failed to send email for {sym}: {mail_e}")
                    except Exception as fleet_e:
                        log.debug(f"[RECONCILER] Error processing fleet table {tbl}: {fleet_e}")

                # 5. Prune stale asset_locks older than 900s (15 min) that have no active position on exchange
                try:
                    if exchange_verified_empty or len(active_exchange_positions) > 0:
                        if live_symbols:
                            placeholders = ",".join("?" for _ in live_symbols)
                            stale_cur = conn.execute(
                                f"DELETE FROM asset_locks WHERE symbol NOT IN ({placeholders}) "
                                f"AND locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect') "
                                f"AND (julianday(?) - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0",
                                (*list(live_symbols), now_iso)
                            )
                        else:
                            stale_cur = conn.execute(
                                "DELETE FROM asset_locks WHERE locked_by IN ('super_bot_grid', 'super_bot_scalp_short', 'swing_architect') "
                                "AND (julianday(?) - julianday(COALESCE(heartbeat, locked_at))) * 86400.0 > 900.0",
                                (now_iso,)
                            )
                        if stale_cur.rowcount and stale_cur.rowcount > 0:
                            log.info(f"[RECONCILER-LOCK-PRUNE] Pruned {stale_cur.rowcount} stale asset_locks (>900s old with no exchange position).")
                except Exception as lk_err:
                    log.debug(f"[RECONCILER-LOCK-PRUNE-NOTE] {lk_err}")

                conn.commit()

            return {
                "mode": "live_bitget_uta",
                "exchange_positions": len(active_exchange_positions),
                "recovered_to_db": recovered_count,
                "pruned_phantoms": pruned_count,
                "balance": bal,
                "synced_at": now_iso
            }
        except Exception as e:
            log.warning(f"[EXCHANGE] Bitget UTA live reconciliation failed: {e}")
            return {"mode": "live_error", "error": str(e)}
        finally:
            if uta:
                try:
                    await uta.close()
                except Exception:
                    pass

    import ccxt.async_support as ccxt
    exchange = ccxt.bitget({
        "apiKey": config.ACTIVE_API_KEY,
        "secret": config.ACTIVE_SECRET_KEY,
        "password": config.ACTIVE_PASSPHRASE,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "adjustForTimeDifference": True}
    })

    try:
        await exchange.load_time_difference()
        positions = await exchange.fetch_positions()
        active_exchange_positions = {}
        for p in positions:
            contracts = float(p.get("contracts", 0.0))
            if abs(contracts) > 0:
                sym = p.get("symbol")
                active_exchange_positions[sym] = {
                    "side": p.get("side"),
                    "contracts": contracts,
                    "entry_price": float(p.get("entryPrice", 0.0)),
                    "unrealized_pnl": float(p.get("unrealizedPnl", 0.0))
                }

        log.info(f"[EXCHANGE] Live Bitget exchange positions found: {len(active_exchange_positions)}")
        return {
            "mode": "live_bitget",
            "exchange_positions": len(active_exchange_positions),
            "synced_at": datetime.now(UTC).isoformat()
        }
    except Exception as e:
        log.warning(f"[EXCHANGE] Live Bitget position fetch failed: {e}")
        return {"mode": "live_error", "error": str(e)}
    finally:
        await exchange.close()

async def reconciliation_main_loop():
    log.info("=" * 60)
    log.info("     SOVEREIGN POSITION & BALANCE RECONCILER INITIALIZED")
    log.info(f"     Monitoring 4 Fleet Tables every {RECONCILIATION_INTERVAL_SECS}s")
    log.info("=" * 60)

    while True:
        try:
            # Check for immediate Operator Force Re-Sync signal from Dashboard
            try:
                with sqlite3.connect(DATA_DB, timeout=3.0) as conn_sig:
                    sig_row = conn_sig.execute("SELECT value FROM system_state WHERE key = 'force_resync'").fetchone()
                    if sig_row and str(sig_row[0]).strip() == "1":
                        log.info("[FORCE-RESYNC] 🚨 Immediate Operator Force Re-Sync signal detected from Dashboard. Triggering audit now.")
                        conn_sig.execute("UPDATE system_state SET value = '0', updated_at = ? WHERE key = 'force_resync'", (datetime.now(UTC).isoformat(),))
                        conn_sig.commit()
            except Exception as sig_err:
                log.debug(f"[FORCE-RESYNC-NOTE] Signal check: {sig_err}")

            # 1. Run local cross-fleet paper audit
            paper_audit = audit_paper_positions()
            
            # 2. Run live exchange audit if live credentials exist
            live_audit = await audit_live_exchange_positions()
            
            # 3. Combine telemetry and publish to system_state
            telemetry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "status": "HEALTHY",
                "paper_audit": paper_audit,
                "live_audit": live_audit
            }
            update_system_state("reconciliation_heartbeat", json.dumps(telemetry))
            log.info(f"[HEARTBEAT] Active: {paper_audit['positions_active']} | Desyncs Healed: {paper_audit['desyncs_healed']} | Status: OK")
            
        except Exception as e:
            log.error(f"Reconciliation cycle exception: {e}")
            
        # Fast responsive sleep loop (checks for force_resync every 1s instead of blocking for full interval)
        for _ in range(int(RECONCILIATION_INTERVAL_SECS)):
            try:
                with sqlite3.connect(DATA_DB, timeout=2.0) as conn_chk:
                    r = conn_chk.execute("SELECT value FROM system_state WHERE key = 'force_resync'").fetchone()
                    if r and str(r[0]).strip() == "1":
                        break
            except Exception:
                pass
            await asyncio.sleep(1.0)

if __name__ == "__main__":
    try:
        asyncio.run(reconciliation_main_loop())
    except KeyboardInterrupt:
        log.info("Reconciliation daemon terminated by user.")
