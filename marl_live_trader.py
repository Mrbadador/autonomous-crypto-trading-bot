# marl_live_trader.py — V15.0 Live Paper-Trading Deployment Loop

import os
import time
import asyncio
import logging
import subprocess
import requests
import sqlite3
import aiohttp
import sys
import dns_fix  # noqa: F401 — patches ccxt.Exchange.open() with ThreadedResolver
from datetime import datetime, UTC
from dotenv import load_dotenv
from swarm_orchestrator import SwarmOrchestrator
from core.overlord import RewardComposer
import config  # Phase 16.9 P0: needed for GLOBAL_MAX_DD_LIMIT in drawdown gate

# ENV CUSTODIAN: Force-load .env from the project root before any I/O or network clients.
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ENV INTEGRITY CHECK
_bitget_key = os.getenv("BITGET_API_KEY", "")
if not _bitget_key:
    print("[MARL_LIVE_TRADER] FATAL: BITGET_API_KEY is not configured in .env. Aborting.", flush=True)
    sys.exit(1)
print(f"[MARL_LIVE_TRADER] ENV OK — BITGET_API_KEY loaded (last 4: ...{_bitget_key[-4:]})", flush=True)

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] LIVE_TRADER: %(message)s")
log = logging.getLogger("live_trader")

_last_known_btc_price = 65000.00

async def get_live_btc_price():
    """
    Fetches real-time Bitcoin price.
    Primary: local Bitget order book state (microstructure_state.json) if < 5s old.
    Secondary: Bitget USDT-FUTURES public ticker API.
    Fallback: Binance ticker API.
    """
    global _last_known_btc_price

    # 1. Fast local Bitget microstructure state
    try:
        import json as _json
        _ms_path = "microstructure_state.json"
        if os.path.exists(_ms_path):
            with open(_ms_path, "r") as _f:
                _ms = _json.load(_f)
                if time.time() - float(_ms.get("timestamp", 0)) < 5.0:
                    _bid = float(_ms.get("best_bid", 0.0) or 0.0)
                    _ask = float(_ms.get("best_ask", 0.0) or 0.0)
                    if _bid > 0 and _ask > 0:
                        _mid = (_bid + _ask) / 2.0
                        _last_known_btc_price = _mid
                        return _mid
    except Exception:
        pass

    # 2. Direct Bitget USDT-FUTURES ticker
    bitget_url = "https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol=BTCUSDT"
    try:
        import socket as _socket
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.ThreadedResolver(),
            family=_socket.AF_INET,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(bitget_url, timeout=aiohttp.ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    ticks = data.get("data", [])
                    if ticks and len(ticks) > 0:
                        raw_p = ticks[0].get("lastPr") or ticks[0].get("markPrice")
                        if raw_p:
                            price = float(raw_p)
                            _last_known_btc_price = price
                            return price
    except Exception:
        pass

    # 3. Bitget fallback
    url = "https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol=BTCUSDT"
    try:
        import socket as _socket
        connector = aiohttp.TCPConnector(
            resolver=aiohttp.ThreadedResolver(),
            family=_socket.AF_INET,
        )
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    entry = data.get("data", [])
                    if entry:
                        raw_p = entry[0].get("lastPr")
                        if raw_p:
                            price = float(raw_p)
                            _last_known_btc_price = price
                            return price
    except Exception as e:
        log.warning(f"All price feeds failed: {e}. Using cached price: ${_last_known_btc_price:,.2f}")
        
    return _last_known_btc_price

def _query_readiness():
    with sqlite3.connect("trading_ledger.db", timeout=15.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        cursor = conn.execute("SELECT signal_readiness FROM market_context ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.50

def _query_regime():
    with sqlite3.connect("trading_data.db", timeout=15.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        cursor = conn.execute("SELECT regime, hurst, adx FROM market_regime ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        return row if row else None


async def run_live_swarm():
    """
    Infinite deployment loop coordinating Oracle updates, real-time prices,
    and policy model forward passes.
    """
    log.info("Starting Swarm Orchestrator Brain...")
    orchestrator = SwarmOrchestrator()
    reward_composer = RewardComposer()
    
    # Inject live exchange or paper exchange depending on staging configuration
    is_live_staging = os.getenv("LIVE_STAGING", "false").lower() == "true" or os.getenv("MODE", "paper").lower() in ("live", "staging")
    
    if is_live_staging:
        import ccxt.async_support as ccxt
        log.info("[BOOT] Initializing live CCXT Bitget Futures client for staging...")
        
        # DNS FIX: Build a dedicated aiohttp session with ThreadedResolver
        import socket as _socket
        _resolver  = aiohttp.ThreadedResolver()
        _connector = aiohttp.TCPConnector(
            resolver=_resolver,
            family=_socket.AF_INET,
            use_dns_cache=True,
            ttl_dns_cache=300,
        )
        _session = aiohttp.ClientSession(connector=_connector)
        
        ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
        ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
        ex_opts = {
            "apiKey": config.ACTIVE_API_KEY,
            "secret": config.ACTIVE_SECRET_KEY,
            "enableRateLimit": True,
            "sandbox": os.getenv("SANDBOX", "true").lower() == "true",
            "options": {
                "defaultType": "swap" if ex_id == "bitget" else "future",
                "adjustForTimeDifference": True,
                "fetchCurrencies": False,
            },
            "timeout": 30000,
            "aiohttp_session": _session,
        }
        if ex_id == "bitget":
            ex_opts["password"] = config.ACTIVE_PASSPHRASE
        staging_exchange = ex_class(ex_opts)
        
        # Enforce ISOLATED margin mode on boot
        log.info("[BOOT] Enforcing ISOLATED margin mode for primary and secondary pairs...")
        # Phase 16.9 P1 Fix: numeric Binance API error code evaluation replaces fragile
        # string-match fallback. Codes: 4046 = no need to change, 4048 = open orders/positions.
        _MARGIN_SAFE_CODES = frozenset({4046, 4048, 40085})
        _MARGIN_SAFE_STRINGS = (
            "margin mode no change", "no need to change margin type",
            "already set to isolated", "already isolated",
            "cannot be changed with open", "unified account mode",
            "classic account api is not supported",
        )
        for symbol in ["BTC/USDT"]:
            try:
                await staging_exchange.set_margin_mode("ISOLATED", symbol)
                log.info(f"[MARGIN-ENFORCEMENT] Programmatically enforced ISOLATED margin mode for {symbol}.")
            except ccxt.ExchangeError as e:
                err_msg = str(e).lower()
                # Extract numeric code from CCXT exception — populated from Binance JSON body
                err_code = getattr(e, "code", None) or getattr(e, "error_code", None)
                try:
                    err_code = int(err_code)
                except (TypeError, ValueError):
                    err_code = None
                is_safe = (err_code in _MARGIN_SAFE_CODES) or any(s in err_msg for s in _MARGIN_SAFE_STRINGS)
                if is_safe:
                    log.info(
                        f"[MARGIN-ENFORCEMENT] {symbol} already operating in ISOLATED mode "
                        f"(API code={err_code}). Proceeding safely."
                    )
                else:
                    log.critical(f"[MARGIN-ENFORCEMENT] TRUE margin conflict for {symbol}: {e} (code={err_code}).")
                    log.critical("[MARGIN-ENFORCEMENT] HALTING SYSTEM INITIALIZATION to protect capital.")
                    await staging_exchange.close()
                    await _session.close()
                    sys.exit(1)
            except (ccxt.NetworkError, asyncio.TimeoutError) as e:
                # Transient network fault during margin assertion — do NOT halt.
                # The exchange will reject orders if the account is actually misconfigured;
                # it will NOT silently misbehave with the wrong margin mode.
                log.warning(
                    f"[MARGIN-ENFORCEMENT] Transient network error during margin check for {symbol}: "
                    f"{type(e).__name__}: {e}. Proceeding with caution."
                )
            except Exception as e:
                log.critical(
                    f"[MARGIN-ENFORCEMENT] Unexpected exception type during margin check for {symbol}: "
                    f"{type(e).__name__}: {e}. HALTING."
                )
                await staging_exchange.close()
                await _session.close()
                sys.exit(1)
                    
        staging_exchange.account_name = "marl_live"
        orchestrator.executor.exchange = staging_exchange
        orchestrator.executor.is_live = True
        paper_ex = None
        log.info("[BOOT] Live staging exchange successfully injected into Swarm Orchestrator.")
    else:
        # Inject live_paper_exchange instance directly into the ExecutorAgent for local execution
        from live_paper_exchange import LivePaperExchange
        _init_bal = float(os.getenv("INITIAL_BALANCE", "58.77"))
        paper_ex = LivePaperExchange(initial_balance=_init_bal, account_name="marl_live")
        orchestrator.executor.exchange = paper_ex
        orchestrator.executor.is_live = False
    
    # Helper to query system state for macro trend checks (F-03 Remediation)
    def query_system_state(key):
        try:
            with sqlite3.connect("trading_data.db", timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=15000;")
                cursor = conn.execute("SELECT value FROM system_state WHERE key=? ORDER BY updated_at DESC LIMIT 1", (key,))
                row = cursor.fetchone()
                return str(row[0]).upper() if row else "UNKNOWN"
        except Exception as e:
            log.warning(f"[MACRO-GUARD] Failed to query system state for {key}: {e}")
            return "UNKNOWN"

    def query_sentiment_metrics():
        try:
            with sqlite3.connect("trading_ledger.db", timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                cursor = conn.execute("SELECT fear_greed_index FROM market_context ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                fg = float(row[0]) if row and row[0] is not None else 50.0
                
                cursor = conn.execute("SELECT spy_health FROM macro_logs ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                spy = float(row[0]) if row and row[0] is not None else 0.0
                
                return fg, spy
        except Exception as e:
            log.warning(f"[MACRO-GUARD] Failed to query sentiment metrics: {e}")
            return 50.0, 0.0

    # Inject Macro Trend Guard by wrapping execute_trade (F-03 Remediation)
    original_execute_trade = orchestrator.executor.execute_trade
    
    async def wrapped_execute_trade(strategy_decision, current_price, reward_weights=None, market_state=None, symbol="BTC/USDT", gamma=0.95):
        action_name = strategy_decision.get("action_name", "HOLD").upper()

        # ── [P0-DD-GATE] CP35-Task02+Task03: Unified Peak-Balance Drawdown Firewall ──────────
        # Standardized to the same peak-balance formula used by EngineState.can_open_new_trade()
        # in state_manager.py:181 to eliminate the prior asymmetry between the two enforcement
        # layers. Formula: dd = ((max_balance - current_balance) / max_balance) * 100.0
        #
        # CP35-Task03: The prior 'paper_ex is not None' guard is REMOVED. In live-staging mode
        # (paper_ex is None) we now query the exchange directly so the gate is NEVER bypassed.
        # ─────────────────────────────────────────────────────────────────────────────────────
        if action_name in ("LONG", "SHORT"):
            try:
                if paper_ex is not None:
                    if getattr(paper_ex, "is_live_execution", False) and getattr(paper_ex, "uta_client", None):
                        try:
                            live_eq = await paper_ex.uta_client.fetch_balance()
                            current_bal = float(live_eq) if live_eq and live_eq > 0 else float(paper_ex.balance)
                        except Exception:
                            current_bal = float(paper_ex.balance)
                    else:
                        # Paper mode: calculate total equity (cash + margin reserved)
                        total_equity = float(paper_ex.balance)
                        for pos in getattr(paper_ex, "positions", {}).values():
                            q = float(pos.get("qty", 0.0) or 0.0)
                            ep = float(pos.get("entry_price", 0.0) or 0.0)
                            lev = float(pos.get("leverage", 1.0) or 1.0)
                            total_equity += (q * ep) / max(1.0, lev)
                        current_bal = total_equity

                    # Peak-balance tracking: track the high-water mark across the session
                    def _query_hist_peak():
                        pks = [137.0]
                        try:
                            with sqlite3.connect("trading_data.db", timeout=5.0) as _c:
                                for _r in _c.execute("SELECT value FROM system_state WHERE key IN ('hwm_balance', 'peak_balance_sniper', 'peak_balance_live')").fetchall():
                                    try:
                                        _v = float(_r[0])
                                        if _v > 0: pks.append(_v)
                                    except Exception: pass
                        except Exception: pass
                        return max(pks)

                    _peak_attr = "_dd_gate_peak_balance"
                    if not hasattr(paper_ex, _peak_attr):
                        setattr(paper_ex, _peak_attr, max(float(paper_ex.initial_balance), _query_hist_peak()))
                    if current_bal > getattr(paper_ex, _peak_attr):
                        setattr(paper_ex, _peak_attr, current_bal)
                    peak_bal = getattr(paper_ex, _peak_attr)
                else:
                    # Live-staging mode: query real wallet balance from exchange (CP35-Task03)
                    # No bypass allowed — the live exchange does NOT enforce portfolio drawdown.
                    _live_ex = orchestrator.executor.exchange
                    if _live_ex is None:
                        log.error("[P0-DD-GATE] No exchange handle available in live-staging mode. Blocking trade as precaution.")
                        return False
                    try:
                        # Support Bitget UTA direct asset query if exchange is Bitget or ccxt fails
                        current_bal = 0.0
                        try:
                            from bitget_uta_client import BitgetUTAClient
                            _u_tmp = BitgetUTAClient()
                            current_bal = await _u_tmp.fetch_balance()
                            await _u_tmp.close()
                        except Exception:
                            pass
                        
                        if current_bal <= 0.0:
                            _acct = await _live_ex.fetch_balance()
                            free_usdt = float(_acct.get("free", {}).get("USDT", 0.0) or _acct.get("USDT", {}).get("free", 0.0))
                            used_usdt = float(_acct.get("used", {}).get("USDT", 0.0) or _acct.get("USDT", {}).get("used", 0.0))
                            total_usdt = float(_acct.get("total", {}).get("USDT", 0.0) or _acct.get("USDT", {}).get("total", 0.0))
                            current_bal = total_usdt if total_usdt > 0.0 else (free_usdt + used_usdt)
                    except Exception as _bal_err:
                        log.error(f"[P0-DD-GATE] Live balance fetch failed: {_bal_err}. Blocking trade as precaution.")
                        return False
                    # Peak-balance tracking for live mode (session-scoped closure variable)
                    _live_peak_attr = "_dd_gate_live_peak"
                    if not hasattr(orchestrator.executor, _live_peak_attr):
                        setattr(orchestrator.executor, _live_peak_attr, max(current_bal, _query_hist_peak()))
                    if current_bal > getattr(orchestrator.executor, _live_peak_attr):
                        setattr(orchestrator.executor, _live_peak_attr, current_bal)
                    peak_bal = getattr(orchestrator.executor, _live_peak_attr)

                # Unified peak-balance drawdown calculation (matches state_manager.py:181)
                if peak_bal > 0:
                    current_dd_pct = ((peak_bal - current_bal) / peak_bal) * 100.0
                    if current_dd_pct >= config.GLOBAL_MAX_DD_LIMIT:
                        log.warning(
                            f"[P0-DD-GATE] MARL trade vetoed: Peak-balance drawdown limit breached. "
                            f"Peak=${peak_bal:,.2f} | Current=${current_bal:,.2f} | "
                            f"DD={current_dd_pct:.2f}% >= limit={config.GLOBAL_MAX_DD_LIMIT:.1f}%. "
                            f"No new {action_name} position opened until capital recovers."
                        )
                        return False
                    else:
                        log.debug(f"[P0-DD-GATE] Gate clear: DD={current_dd_pct:.2f}% < {config.GLOBAL_MAX_DD_LIMIT:.1f}%.")
            except Exception as _dd_err:
                log.error(f"[P0-DD-GATE] Unhandled exception in drawdown gate: {_dd_err}. Blocking trade as precaution.")
                return False

        # Check if we are holding an open position that contradicts this AI signal!
        def _get_active_pos():
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    conn.row_factory = sqlite3.Row
                    return conn.execute("SELECT * FROM marl_live_open_positions WHERE symbol LIKE '%BTC%'").fetchone()
            except Exception:
                return None

        active_pos = await asyncio.to_thread(_get_active_pos)
        if active_pos:
            curr_side = str(active_pos["side"]).lower()
            # If AI signals opposite direction, this is an explicit AI EXIT command
            if (curr_side == "long" and action_name == "SHORT") or (curr_side == "short" and action_name == "LONG"):
                log.info(f"[AI-SIGNAL-EXIT] [TARGET] MARL AI selected {action_name} while holding {curr_side.upper()}! Respecting AI exit signal -> Closing {curr_side.upper()} to CASH.")
                from bitget_uta_client import BitgetUTAClient
                _uclient = BitgetUTAClient()
                try:
                    await _uclient.close_position_order(symbol="BTC/USDT", side=curr_side, qty=float(active_pos["qty"]))
                    now_iso = datetime.now(UTC).isoformat()
                    with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                        exit_p = current_price
                        entry_p = float(active_pos["entry_price"])
                        qty_val = float(active_pos["qty"])
                        net_pnl = (exit_p - entry_p) * qty_val if curr_side == "long" else (entry_p - exit_p) * qty_val
                        conn.execute("""
                            INSERT INTO marl_live_trades (timestamp, symbol, side, type, entry_price, exit_price, qty, pnl, balance, regime, phase, quality, notes, status, ai_rationale, slippage_bps, action)
                            VALUES (?, 'BTC/USDT', ?, 'ai_policy_exit', ?, ?, ?, ?, 0.0, 'MARL_SWARM', 'POLICY_REVERSAL', 0.0, 'AI Policy Exit', 'closed', 'AI Policy selected opposite action', 0.0, 'ai_exit')
                        """, (now_iso, curr_side, entry_p, exit_p, qty_val, net_pnl))
                        conn.execute("DELETE FROM marl_live_open_positions WHERE symbol = 'BTC/USDT'")
                        conn.commit()
                    log.info(f"[AI-SIGNAL-EXIT] Successfully closed {curr_side.upper()} to CASH based on AI policy decision.")

                    try:
                        from email_dispatcher import send_trade_log_async
                        act_dict = dict(active_pos) if active_pos else {}
                        close_ctx = {
                            "engine": "MARL Swarm Agent (15m)",
                            "side": curr_side.upper(),
                            "qty": qty_val,
                            "entry_price": entry_p,
                            "price": exit_p,
                            "pnl": net_pnl,
                            "leverage": float(act_dict.get("leverage", 5.0) or 5.0),
                            "close_trigger": "AI_POLICY_REVERSAL",
                            "close_reason": "MARL Policy selected opposing action. Closed position to cash."
                        }
                        asyncio.create_task(send_trade_log_async(symbol="BTC/USDT", side=curr_side.upper(), price=exit_p, pnl=net_pnl, status="CLOSED", trade_context=close_ctx))
                    except Exception as me:
                        log.warning(f"[EMAIL-NOTE] MARL exit email note: {me}")
                except Exception as ce:
                    log.error(f"[AI-SIGNAL-EXIT] Error closing position on AI signal: {ce}")
                finally:
                    await _uclient.close()
                # BUG-05 FIX: Return False after AI-signal exit to prevent same-cycle re-entry.
                # The next 5-minute cycle will evaluate the new signal cleanly.
                return False

        # Strict Macro Directional Alignment Gate (Option B)
        macro_trend = await asyncio.to_thread(query_system_state, "btc_macro_trend")
        if action_name == "LONG" and macro_trend == "BEARISH":
            log.warning("[MACRO-DIRECTION-LOCK] VETO: LONG trade suppressed because Macro Trend is BEARISH.")
            return False
        if action_name == "SHORT" and macro_trend == "BULLISH":
            log.warning("[MACRO-DIRECTION-LOCK] VETO: SHORT trade suppressed because Macro Trend is BULLISH.")
            return False

        # 1.2 Hurst Exponent & ADX Chop Filter Gate
        reg_data = await asyncio.to_thread(_query_regime)
        if reg_data:
            r_label, r_hurst, r_adx = str(reg_data[0] or "").upper(), float(reg_data[1] or 0.5), float(reg_data[2] or 20.0)
            hurst_floor = getattr(config, "HURST_TREND_FLOOR", 0.52)
            adx_floor = getattr(config, "ADX_CHOP_FLOOR", 15.0)
            if "CHOP" in r_label or r_hurst < hurst_floor or r_adx < adx_floor:
                log.warning(f"[HURST-CHOP-GATE] VETO: MARL {action_name} suppressed in CHOP regime (Regime={r_label}, Hurst={r_hurst:.3f} < {hurst_floor}, ADX={r_adx:.1f} < {adx_floor}).")
                return False

        # 7.4 Macro Economic News Event Gate
        news_active = await asyncio.to_thread(query_system_state, "news_event_active")
        if str(news_active).lower() in ("1", "true"):
            news_reason = await asyncio.to_thread(query_system_state, "news_event_reason")
            log.warning(f"[NEWS-CALENDAR-VETO] MARL {action_name} suppressed due to active macro event: {news_reason}")
            return False

        # Macro Sentiment Override (Emergency Brakes)
        fg, spy = await asyncio.to_thread(query_sentiment_metrics)
        is_extreme_panic = (fg < 25.0) or (spy < -0.6)
        
        if action_name == "LONG" and is_extreme_panic:
            log.critical(f"[EMERGENCY-BRAKE] LONG trade vetoed! Macro Sentiment is in EXTREME PANIC. F&G={fg:.1f}, SPY_Health={spy:.2f}. Forcing HOLD/CASH state.")
            return False

        exec_res = await original_execute_trade(strategy_decision, current_price, reward_weights, market_state, symbol, gamma)
        if exec_res and action_name in ("LONG", "SHORT"):
            try:
                from email_dispatcher import send_trade_log_async
                pos_row = await asyncio.to_thread(_get_active_pos)
                pos_data = dict(pos_row) if pos_row else {}
                entry_p = float(pos_data.get("entry_price", current_price) or current_price)
                qty_val = float(pos_data.get("qty", 0.0) or 0.0)
                sl_val = float(pos_data.get("stop_loss", 0.0) or 0.0)
                tp_val = float(pos_data.get("take_profit", 0.0) or 0.0)
                lev_val = float(pos_data.get("leverage", 5.0) or 5.0)
                conf_score = float(strategy_decision.get("confidence", 0.0) or 0.0)

                trade_ctx = {
                    "engine": "MARL Swarm Agent (15m)",
                    "side": action_name,
                    "symbol": symbol,
                    "price": entry_p,
                    "qty": qty_val,
                    "stop_loss": sl_val,
                    "take_profit": tp_val,
                    "conviction": f"{round(conf_score * 10)}/10",
                    "quality": conf_score * 100.0,
                    "leverage": lev_val,
                    "ai_reasoning": f"MARL Multi-Agent Swarm consensus entry. Confidence={conf_score:.1%}, Gamma={gamma:.2f}."
                }
                asyncio.create_task(send_trade_log_async(
                    symbol=symbol,
                    side=action_name,
                    price=entry_p,
                    status="OPENED",
                    trade_context=trade_ctx
                ))
                log.info(f"[MARL-EMAIL] Dispatched trade OPEN alert for {symbol} {action_name} @ ${entry_p:,.2f}")
            except Exception as _em_err:
                log.warning(f"[EMAIL-NOTE] Failed to dispatch MARL open email: {_em_err}")
        return exec_res

    orchestrator.executor.execute_trade = wrapped_execute_trade
    
    interval_seconds = 300  # 5 minutes
    
    try:
        while True:
            print("\n" + "#" * 60)
            print("          >>> DISPATCHING LIVE SWARM AUDIT CYCLE <<<")
            print("#" * 60)
            
            # Step 1: Read Live Market Metrics (Managed continuously by persistent live_oracle daemon)
            log.info("Step 1: Reading real-time market context from Live Oracle ledger...")
                
            # Step 2: Retrieve Live Price
            log.info("Step 2: Fetching live asset price...")
            current_btc_price = await get_live_btc_price()
            
            # Step 2.1: Monitor active positions for SL/TP exit & profit management (Live & Paper)
            try:
                def _get_marl_pos():
                    try:
                        with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                            conn.row_factory = sqlite3.Row
                            return conn.execute("SELECT * FROM marl_live_open_positions WHERE symbol LIKE '%BTC%'").fetchone()
                    except Exception:
                        return None

                pos_row = await asyncio.to_thread(_get_marl_pos)
                if pos_row and not pos_row["is_closing"]:
                    pos = dict(pos_row)
                    side = pos["side"]
                    entry_p = float(pos.get("entry_price", 0.0))
                    peak_p = float(pos.get("peak_price", entry_p) or entry_p)
                    
                    if side == "long":
                        peak_p = max(peak_p, current_btc_price)
                    else:
                        peak_p = min(peak_p, current_btc_price)
                    pos["peak_price"] = peak_p

                    p_dist = (current_btc_price - entry_p) if side == "long" else (entry_p - current_btc_price)
                    fee_buf_pct = float(getattr(config, "BREAKEVEN_FEE_BUFFER_PCT", 0.001))
                    fee_buffer = entry_p * fee_buf_pct
                    be_sl = (entry_p + fee_buffer) if side == "long" else (entry_p - fee_buffer)

                    # Query ATR
                    atr_val = None
                    try:
                        with sqlite3.connect("trading_data.db", timeout=5.0) as _conn:
                            _cur = _conn.execute("SELECT atr FROM market_regime WHERE atr IS NOT NULL ORDER BY id DESC LIMIT 1")
                            _ar = _cur.fetchone()
                            if _ar and _ar[0] and float(_ar[0]) > 0:
                                atr_val = float(_ar[0])
                    except Exception:
                        pass

                    # 1. Early Profit Breakeven Ratchet (+1.0x ATR gain)
                    be_mult = float(getattr(config, "EARLY_BE_ATR_MULT", 1.0))
                    if atr_val and atr_val > 0 and p_dist >= be_mult * atr_val:
                        if (side == "long" and be_sl > pos["stop_loss"]) or (side == "short" and be_sl < pos["stop_loss"]):
                            pos["stop_loss"] = be_sl
                            log.info(f"[MARL-EARLY-BE] [SHIELD] BTC/USDT reached +{be_mult:.1f}x ATR gain (+${p_dist:,.2f}). Ratcheting SL to Breakeven (${be_sl:,.2f})!")

                    # 2. High-Water Trailing Stop (Chandelier Exit at +1.5x ATR)
                    if atr_val and atr_val > 0 and p_dist >= 1.5 * atr_val:
                        trail_mult = 1.5
                        trail_sl = (peak_p - (trail_mult * atr_val)) if side == "long" else (peak_p + (trail_mult * atr_val))
                        if (side == "long" and trail_sl > pos["stop_loss"]) or (side == "short" and trail_sl < pos["stop_loss"]):
                            pos["stop_loss"] = trail_sl
                            log.info(f"[MARL-TRAILING-STOP] [ROCKET] High-Water Trailing Stop ratcheted to ${trail_sl:,.2f} (Peak: ${peak_p:,.2f}). Locking in profit!")

                    # Sync updated peak and SL to DB
                    def _sync_pos_db():
                        with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                            conn.execute("UPDATE marl_live_open_positions SET stop_loss = ?, peak_price = ? WHERE symbol = ?", (pos["stop_loss"], pos["peak_price"], pos["symbol"]))
                            conn.commit()
                    await asyncio.to_thread(_sync_pos_db)

                    # Check SL / TP
                    is_sl = (side == "long" and current_btc_price <= pos["stop_loss"]) or (side == "short" and current_btc_price >= pos["stop_loss"])
                    is_tp = (pos.get("take_profit", 0) > 0) and (
                        (side == "long" and current_btc_price >= pos["take_profit"]) or
                        (side == "short" and current_btc_price <= pos["take_profit"])
                    )

                    if is_sl or is_tp:
                        close_type = "tp_hit" if is_tp else "sl_hit"
                        log.info(f"[marl_live] Exit triggered for BTC/USDT. Type: {close_type} | Price: ${current_btc_price:,.2f} (SL: ${pos['stop_loss']:,.2f}, TP: ${pos['take_profit']:,.2f})")
                        
                        # BUG-10 FIX: Set is_closing flag BEFORE placing close order to prevent
                        # a duplicate close order if the 5-minute loop fires again while in-flight.
                        def _set_is_closing():
                            try:
                                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                                    conn.execute("UPDATE marl_live_open_positions SET is_closing=1 WHERE symbol=?", (pos["symbol"],))
                                    conn.commit()
                            except Exception:
                                pass
                        await asyncio.to_thread(_set_is_closing)

                        from bitget_uta_client import BitgetUTAClient
                        _ucl = BitgetUTAClient()
                        try:
                            await _ucl.close_position_order(symbol="BTC/USDT", side=side, qty=float(pos["qty"]))
                            now_iso = datetime.now(UTC).isoformat()
                            def _finalize_live_exit():
                                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                                    pnl_val = (current_btc_price - entry_p) * float(pos["qty"]) if side == "long" else (entry_p - current_btc_price) * float(pos["qty"])
                                    conn.execute("""
                                        INSERT INTO marl_live_trades (timestamp, symbol, side, type, entry_price, exit_price, qty, pnl, balance, regime, phase, quality, notes, status, ai_rationale, slippage_bps, action)
                                        VALUES (?, 'BTC/USDT', ?, ?, ?, ?, ?, ?, 0.0, 'MARL_SWARM', 'EXIT', 0.0, '', 'closed', ?, 0.0, ?)
                                    """, (now_iso, side, close_type, entry_p, current_btc_price, float(pos["qty"]), pnl_val, f"MARL {close_type}", close_type))
                                    conn.execute("DELETE FROM marl_live_open_positions WHERE symbol = ?", (pos["symbol"],))
                                    conn.commit()
                            await asyncio.to_thread(_finalize_live_exit)

                            try:
                                from email_dispatcher import send_trade_log_async
                                close_ctx = {
                                    "engine": "MARL Swarm Agent (15m)",
                                    "side": side.upper(),
                                    "qty": float(pos["qty"]),
                                    "entry_price": entry_p,
                                    "price": current_btc_price,
                                    "pnl": pnl_val,
                                    "leverage": float(pos.get("leverage", 3.0)),
                                    "close_trigger": close_type.upper(),
                                    "close_reason": f"MARL {close_type.upper()} triggered at ${current_btc_price:,.2f}."
                                }
                                asyncio.create_task(send_trade_log_async(symbol="BTC/USDT", side=side.upper(), price=current_btc_price, pnl=pnl_val, status="CLOSED", trade_context=close_ctx))
                            except Exception as me:
                                log.warning(f"[EMAIL-NOTE] MARL exit email note: {me}")
                        except Exception as live_exit_err:
                            log.error(f"[marl_live] Error executing exit: {live_exit_err}")
                        finally:
                            await _ucl.close()
            except Exception as pe_err:
                log.debug(f"[MARL-MONITOR-NOTE] {pe_err}")
            
            # Step 2.5: Extract Overlord Information-Theoretic Regime Signals
            log.info("Step 2.5: Querying Overlord Signal Readiness from state ledger...")
            signal_readiness = 0.50
            try:
                signal_readiness = await asyncio.to_thread(_query_readiness)
            except Exception as e:
                log.warning(f"Failed to read signal_readiness from SQLite: {e}. Defaulting to 0.50.")
            
            # Implement Kimi's Overlord Preset Mapping Logic
            if signal_readiness > 0.70:
                regime_name = "TRENDING"
                template_key = "trending"
            elif signal_readiness < 0.30:
                regime_name = "MEAN_REV"
                template_key = "mean_rev"
            else:
                regime_name = "NEUTRAL"
                template_key = "low_vol"
                
            reward_weights = reward_composer.regime_template(template_key)
            w_list = reward_weights.tolist()
            
            log.info(f"[OVERLORD] Signal Readiness: {signal_readiness:.2f} -> Regime: {regime_name}. Sub-Swarm Weights: PnL: {w_list[0]:.2f} | Drawdown: {w_list[1]:.2f} | Sharpe: {w_list[2]:.2f}")
            
            # Step 2.6: Query Market Regime from trading_data.db and calculate dynamic gamma
            log.info("Step 2.6: Querying current market regime from trading_data.db...")
            regime = "NEUTRAL"
            hurst = 0.5
            adx = 20.0
            try:
                regime_row = await asyncio.to_thread(_query_regime)
                if regime_row:
                    regime = str(regime_row[0]).upper()
                    if regime_row[1] is not None:
                        hurst = float(regime_row[1])
                    if regime_row[2] is not None:
                        adx = float(regime_row[2])
            except Exception as e:
                log.warning(f"Failed to query market regime from SQLite: {e}. Defaulting to NEUTRAL.")

            global_risk_cap = 0.05
            if regime in ("TRENDING", "TREND"):
                gamma = 0.99
            elif regime in ("CHOP", "RANGING"):
                gamma = 0.90
                global_risk_cap = 0.03
            else:
                gamma = 0.95

            log.info(f"[REGIME] Hurst={hurst:.4f} | ADX={adx:.2f} | Regime={regime} -> Dynamic Gamma: {gamma} | Risk Cap: {global_risk_cap * 100:.2f}%")

            # Update the executor's maximum risk cap ceiling limit dynamically
            orchestrator.executor.max_cap = global_risk_cap

            if paper_ex is not None:
                try:
                    await paper_ex.update_system_state("marl_live_gamma", gamma)
                    await paper_ex.update_system_state("marl_live_risk_cap", global_risk_cap)
                except Exception as ex:
                    log.error(f"Failed to persist gamma or risk cap to system state: {ex}")

            # Step 3: Execute policy network inference and check transaction gating
            log.info("Step 3: Evaluating active policy decisions with Overlord weights...")
            # Hot-reload policy weights if updated on disk
            try:
                import torch
                for _wpath in ["strategist_weights_v28.pth", "strategist_weights_v27.pth", "policy_model.pth"]:
                    if os.path.exists(_wpath):
                        _mtime = os.path.getmtime(_wpath)
                        _attr = f"_last_mtime_{_wpath}"
                        if getattr(orchestrator.strategist, _attr, 0) != _mtime:
                            orchestrator.strategist.policy.load_state_dict(
                                torch.load(_wpath, map_location=torch.device("cpu"), weights_only=True)
                            )
                            setattr(orchestrator.strategist, _attr, _mtime)
                            log.info(f"[HOT-RELOAD] Loaded fresh neural weights from: {_wpath}")
                            break
            except Exception as _re_err:
                log.warning(f"[HOT-RELOAD] Weight reload check failed: {_re_err}")

            await orchestrator.run_cycle(current_btc_price, reward_weights=reward_weights, gamma=gamma)
            
            # Step 4: Sleep until next interval
            log.info(f"Step 4: Swarm heartbeat successful. Sleeping for {interval_seconds // 60} minutes...")
            await asyncio.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("          !!! LIVE SWARM GRACEFULLY TERMINATED !!!")
        print("=" * 60 + "\n")
        log.info("Graceful shutdown complete. Active positions are preserved.")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      LAUNCHING MARL V15.0 LIVE PAPER-TRADING SYSTEM")
    print("=" * 60 + "\n")
    
    asyncio.run(run_live_swarm())
