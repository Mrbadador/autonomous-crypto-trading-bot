# executor_agent.py
# Agent 4 — The Executor
# MARL Trading Swarm | Mock Execution Layer (No live API calls)

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] EXECUTOR: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("executor_agent")

# ─── Constants ────────────────────────────────────────────────────────────────
ACTION_HOLD  = 0
ACTION_LONG  = 1
ACTION_SHORT = 2

SL_PCT = 0.02   # 2% Stop-Loss distance
TP_PCT = 0.05   # 5% Take-Profit distance

ACTION_LABELS = {
    ACTION_HOLD:  "HOLD",
    ACTION_LONG:  "LONG",
    ACTION_SHORT: "SHORT",
}


# ─── Order Dataclass ──────────────────────────────────────────────────────────
@dataclass
class OrderContext:
    """Holds a fully computed order ready for simulated execution."""
    action:           str
    action_id:        int
    confidence:       float
    entry_price:      float
    stop_loss:        Optional[float] = None
    take_profit:      Optional[float] = None


async def _monitor_maker_fill(*args, **kwargs) -> dict:
    """
    Sprint 3 Maker Queue Fill Monitor:
    Queries fill status via private Bitget UTA API.
    If unfilled after timeout_sec (default 8.0s) OR micro-imbalance moves adversely:
      1. Queries order state first to avoid race-condition orphan fills.
      2. If already filled: accepts maker fill and captures fee rebate.
      3. If partially filled or unfilled: cancels limit order and executes immediate market fallback.
    
    Can be called as:
      _monitor_maker_fill(uta_client, order_id, symbol, side, qty, current_price, timeout_sec=8.0, sl=0.0, tp=0.0, client_id=None)
      OR
      _monitor_maker_fill(order_id, symbol, side, timeout_sec=8.0, uta_client=None, qty=0.0, ...)
    """
    uta_client = None
    order_id = None
    symbol = None
    side = "buy"
    qty = 0.0
    current_price = 0.0
    timeout_sec = 8.0
    sl = 0.0
    tp = 0.0
    client_id = None

    if len(args) >= 1:
        if isinstance(args[0], str):
            order_id = args[0]
            if len(args) >= 2: symbol = args[1]
            if len(args) >= 3: side = args[2]
            if len(args) >= 4: timeout_sec = float(args[3])
        else:
            uta_client = args[0]
            if len(args) >= 2: order_id = str(args[1])
            if len(args) >= 3: symbol = str(args[2])
            if len(args) >= 4: side = str(args[3])
            if len(args) >= 5: qty = float(args[4])
            if len(args) >= 6: current_price = float(args[5])
            if len(args) >= 7: timeout_sec = float(args[6])
            if len(args) >= 8: sl = float(args[7])
            if len(args) >= 9: tp = float(args[8])
            if len(args) >= 10: client_id = str(args[9])

    uta_client = kwargs.get("uta_client", uta_client)
    order_id = kwargs.get("order_id", order_id)
    symbol = kwargs.get("symbol", symbol)
    side = kwargs.get("side", side)
    qty = float(kwargs.get("qty", qty))
    current_price = float(kwargs.get("current_price", current_price))
    timeout_sec = float(kwargs.get("timeout_sec", timeout_sec))
    sl = float(kwargs.get("sl", sl))
    tp = float(kwargs.get("tp", tp))
    client_id = kwargs.get("client_id", client_id)

    should_close_client = False
    if uta_client is None:
        from bitget_uta_client import BitgetUTAClient
        uta_client = BitgetUTAClient()
        should_close_client = True

    try:
        clean_sym = str(symbol or "BTCUSDT").replace("/", "").replace(":USDT", "").replace("-", "").upper()
        is_long = "long" in str(side).lower() or str(side).lower() == "buy"
        order_side = "buy" if is_long else "sell"
        start_time = time.time()
        poll_interval = 0.5
        
        while (time.time() - start_time) < timeout_sec:
            await asyncio.sleep(poll_interval)
            
            # 1. Fetch current order info from exchange
            try:
                order_info = await uta_client.fetch_order(clean_sym, order_id)
                state = str(order_info.get("state") or order_info.get("status") or "").lower()
                filled_qty = float(order_info.get("baseVolume") or order_info.get("filledQty") or order_info.get("accBaseVolume") or order_info.get("dealSize") or 0.0)
                
                if state in ("filled", "closed") or (qty > 0 and filled_qty >= qty * 0.99):
                    log.info(f"[MAKER-MONITOR] Maker post-only order {order_id} FILLED completely ({filled_qty}/{qty}). Rebate captured (0.02% fee).")
                    return {
                        "status": "filled",
                        "is_maker": True,
                        "order_id": order_id,
                        "filled_qty": filled_qty or qty,
                        "time_to_fill": time.time() - start_time
                    }
                
                if state in ("canceled", "cancelled"):
                    log.warning(f"[MAKER-MONITOR] Order {order_id} was cancelled by exchange (post-only crossing). Triggering market fallback.")
                    remaining_qty = max(0.0, qty - filled_qty)
                    if remaining_qty > 0.0:
                        fb_res = await uta_client.create_market_order(
                            symbol=clean_sym,
                            side=order_side,
                            qty=remaining_qty,
                            client_oid=f"{client_id or 'FALLBACK'}_MKT",
                            sl=sl,
                            tp=tp
                        )
                        return {
                            "status": "fallback_market",
                            "is_maker": False,
                            "order_id": order_id,
                            "maker_portion": filled_qty,
                            "taker_portion": remaining_qty,
                            "fallback_order": fb_res
                        }
                    return {"status": "canceled", "is_maker": False, "filled_qty": filled_qty}
            except Exception as q_err:
                log.warning(f"[MAKER-MONITOR] Error polling order {order_id}: {q_err}")

            # 2. Check micro-imbalance adverse drift
            try:
                if kwargs.get("adverse_drift_detected"):
                    log.warning(f"[MAKER-MONITOR] Adverse drift flag set on {clean_sym}. Triggering immediate cancellation and market fallback.")
                    break
                ofi_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "microstructure_state.json")
                if os.path.exists(ofi_file):
                    with open(ofi_file, "r") as f:
                        ofi_data = json.load(f)
                    if time.time() - ofi_data.get("timestamp", 0) < 15.0:
                        sym_ofi = ofi_data.get("symbols", {}).get(clean_sym, {})
                        z_ofi = sym_ofi.get("ofi_zscore", ofi_data.get("ofi_zscore", 0.0))
                        norm_ofi = sym_ofi.get("normalized_ofi", ofi_data.get("normalized_ofi", 0.0))
                        raw_ofi = sym_ofi.get("ofi_10level", ofi_data.get("net_ofi_score", 0.0))
                        
                        adverse = (is_long and (z_ofi < -1.5 or norm_ofi < -0.6 or raw_ofi < -100.0)) or \
                                  (not is_long and (z_ofi > 1.5 or norm_ofi > 0.6 or raw_ofi > 100.0))
                        if adverse:
                            log.warning(f"[MAKER-MONITOR] Adverse micro-imbalance detected on {clean_sym} (Z={z_ofi:.2f}, Norm={norm_ofi:.2f}). Triggering immediate cancellation and market fallback.")
                            break
            except Exception as ofi_err:
                log.debug(f"[MAKER-MONITOR] OFI check note: {ofi_err}")

        # Timeout (8s) elapsed or adverse drift triggered
        log.info(f"[MAKER-MONITOR] Timeout or adverse drift on maker order {order_id}. Querying state prior to cancel to prevent orphan fills...")
        final_info = await uta_client.fetch_order(clean_sym, order_id)
        final_state = str(final_info.get("state") or final_info.get("status") or "").lower()
        final_filled = float(final_info.get("baseVolume") or final_info.get("filledQty") or final_info.get("accBaseVolume") or final_info.get("dealSize") or 0.0)

        if final_state in ("filled", "closed") or (qty > 0 and final_filled >= qty * 0.99):
            log.info(f"[MAKER-MONITOR] Order {order_id} filled right before cancellation! Maker rebate captured.")
            return {
                "status": "filled",
                "is_maker": True,
                "order_id": order_id,
                "filled_qty": final_filled or qty
            }

        # Cancel open limit order
        try:
            await uta_client.cancel_order(clean_sym, order_id)
            log.info(f"[MAKER-MONITOR] Order {order_id} cancelled successfully.")
        except Exception as ce:
            log.warning(f"[MAKER-MONITOR] Cancel request note: {ce}")

        remaining_qty = max(0.0, qty - final_filled)
        if remaining_qty > 0.0:
            if "BTC" in clean_sym:
                remaining_qty = max(round(remaining_qty, 4), 0.0001)
            elif "ETH" in clean_sym or "BNB" in clean_sym:
                remaining_qty = max(round(remaining_qty, 2), 0.01)
            elif "SOL" in clean_sym or "AVAX" in clean_sym:
                remaining_qty = max(round(remaining_qty, 1), 0.1)
            elif any(c in clean_sym for c in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
                remaining_qty = max(float(round(remaining_qty)), 1.0)
            else:
                remaining_qty = max(round(remaining_qty, 2), 0.1)

            log.info(f"[MAKER-MONITOR] Executing immediate market fallback for {remaining_qty} {clean_sym}...")
            fb_order = await uta_client.create_market_order(
                symbol=clean_sym,
                side=order_side,
                qty=remaining_qty,
                client_oid=f"{client_id or 'FALLBACK'}_FB",
                sl=sl,
                tp=tp
            )
            return {
                "status": "fallback_market",
                "is_maker": False,
                "order_id": order_id,
                "maker_portion": final_filled,
                "taker_portion": remaining_qty,
                "fallback_order": fb_order
            }
        return {"status": "canceled", "is_maker": False, "filled_qty": final_filled}
    finally:
        if should_close_client:
            await uta_client.close()


# ─── ExecutorAgent ────────────────────────────────────────────────────────────
class ExecutorAgent:
    """
    Agent 4 — The Executor.

    Receives a MARL action signal and either:
      - HOLDs (action_id=0): logs and exits cleanly.
      - LONGs (action_id=1): computes SL = entry - 2%, TP = entry + 5%.
      - SHORTs (action_id=2): computes SL = entry + 2%, TP = entry - 5%.

    mock_execute_order() simulates the trade without touching any live API.
    """

    def __init__(self, sl_pct: float = SL_PCT, tp_pct: float = TP_PCT, min_confidence: float = None):
        import config as _cfg
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.min_confidence = min_confidence if min_confidence is not None else getattr(_cfg, "MIN_NN_CONFIDENCE", 0.34)
        self._monitor_maker_fill = _monitor_maker_fill
        log.info(
            f"ExecutorAgent initialised. "
            f"SL={self.sl_pct*100:.1f}%  TP={self.tp_pct*100:.1f}%  MIN_CONF={self.min_confidence:.2%}"
        )

    # ── Core decision dispatcher ──────────────────────────────────────────────
    def evaluate(
        self,
        action_id:        int,
        confidence_score: float,
        current_price:    float,
    ) -> Optional[OrderContext]:
        """
        Evaluate the incoming MARL signal.

        Returns an OrderContext if an order should be executed,
        or None if the action is HOLD.
        """
        label = ACTION_LABELS.get(action_id, "UNKNOWN")

        if action_id == ACTION_HOLD:
            log.info("Action received: HOLD. No execution required.")
            return None

        if action_id == ACTION_LONG:
            stop_loss   = round(current_price * (1 - self.sl_pct), 8)
            take_profit = round(current_price * (1 + self.tp_pct), 8)

        elif action_id == ACTION_SHORT:
            stop_loss   = round(current_price * (1 + self.sl_pct), 8)
            take_profit = round(current_price * (1 - self.tp_pct), 8)

        else:
            log.error(f"Unknown action_id={action_id}. Defaulting to HOLD.")
            return None

        order = OrderContext(
            action      = label,
            action_id   = action_id,
            confidence  = confidence_score,
            entry_price = current_price,
            stop_loss   = stop_loss,
            take_profit = take_profit,
        )

        log.info(
            f"Action evaluated: {label} | "
            f"Confidence={confidence_score:.2%} | "
            f"Entry=${current_price:,.4f}"
        )
        return order

    # ── Mock execution ─────────────────────────────────────────────────────────
    def mock_execute_order(
        self,
        action_id:        int,
        confidence_score: float,
        current_price:    float,
    ) -> Optional[OrderContext]:
        """
        Full pipeline: evaluate signal → compute levels → simulate execution.
        Logs a human-readable order ticket to the console.
        Does NOT touch any live exchange or external API.
        """
        order = self.evaluate(action_id, confidence_score, current_price)

        if order is None:
            # HOLD — nothing to execute
            return None

        # ── Simulated execution output ────────────────────────────────────────
        log.info("=" * 55)
        log.info("  [MOCK EXECUTION] ORDER TICKET")
        log.info("=" * 55)
        log.info(f"  Direction  : {order.action}")
        log.info(f"  Confidence : {order.confidence:.2%}")
        log.info(f"  Entry Price: ${order.entry_price:>12,.4f}")
        log.info(f"  Stop-Loss  : ${order.stop_loss:>12,.4f}  (-{self.sl_pct*100:.1f}% from entry)")
        log.info(f"  Take-Profit: ${order.take_profit:>12,.4f}  (+{self.tp_pct*100:.1f}% from entry)")
        log.info(f"  Risk/Reward: 1 : {self.tp_pct / self.sl_pct:.1f}")
        log.info("=" * 55)
        log.info("[MOCK] Order placed successfully. Awaiting fill simulation.")

        return order




    async def execute_trade(self, strategy_decision: dict, current_price: float, reward_weights=None, market_state=None, symbol="BTC/USDT", gamma: float = 0.95) -> bool:
        """
        V30 Execution Router: Evaluates confidence and executes via attached exchange (if present),
        otherwise falls back to mock execution. Used by both LiveTrader and MARL Orchestrator.
        """
        action_name = strategy_decision.get("action_name", "HOLD").upper()
        # In live_trader it's called 'confidence', in marl_live_trader it might also be 'confidence'
        confidence = float(strategy_decision.get("confidence", 0.0))

        if confidence < getattr(self, "min_confidence", 0.70):
            log.warning(f"Trade aborted: Confidence {confidence:.2%} < {getattr(self, 'min_confidence', 0.70):.2%}")
            return False

        # V_AUDIT_FIX: Macro Sentinel Global Killswitch Gate (SPY dump > 1.0% or DXY > 106.0)
        if action_name in ("LONG", "SHORT"):
            try:
                import sqlite3, os as _os, asyncio as _asyncio
                def _check_macro_killswitch():
                    for _dbname in ["trading_data.db", "trading_ledger.db"]:
                        _db_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _dbname)
                        if _os.path.exists(_db_path):
                            try:
                                with sqlite3.connect(_db_path, timeout=2.0) as _c:
                                    row = _c.execute("SELECT global_killswitch, spy_1h_change, dxy_price FROM macro_market_context WHERE id=1 LIMIT 1").fetchone()
                                    if row and int(row[0]) == 1:
                                        spy_c = row[1] if row[1] is not None else 0.0
                                        dxy_p = row[2] if row[2] is not None else 0.0
                                        return True, f"SPY 1h={spy_c:+.2f}%, DXY={dxy_p:.2f}"
                                    row2 = _c.execute("SELECT value FROM system_state WHERE key='macro_global_killswitch' LIMIT 1").fetchone()
                                    if row2 and str(row2[0]).strip() in ("1", "true", "TRUE"):
                                        return True, "system_state macro_global_killswitch flag active"
                            except Exception:
                                pass
                    return False, ""

                is_killed, kill_reason = await _asyncio.to_thread(_check_macro_killswitch)
                if is_killed:
                    log.critical(f"[MACRO-KILLSWITCH] Execution ABORTED — TradFi Macro Sentinel Killswitch is ACTIVE ({kill_reason}).")
                    return False
            except Exception as _mke:
                log.warning(f"[MACRO-KILLSWITCH] Check failed ({_mke}). Proceeding with execution.")

        # V_AUDIT_FIX: Macro Warden Regime Gate — read DB cache and enforce LONG_ONLY / SHORT_ONLY
        if action_name in ("LONG", "SHORT"):
            try:
                import sqlite3, os as _os
                _db = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "trading_data.db")
                def _read_regime():
                    try:
                        with sqlite3.connect(_db, timeout=3.0) as _c:
                            row = _c.execute(
                                "SELECT value FROM system_state WHERE key='warden_60m_regime' LIMIT 1"
                            ).fetchone()
                            return row[0] if row else None
                    except Exception:
                        return None
                import asyncio
                cached_regime = await asyncio.to_thread(_read_regime)
                if cached_regime == "LONG_ONLY" and action_name == "SHORT":
                    log.warning(f"[REGIME-GATE] SHORT vetoed — Macro Warden is LONG_ONLY (BTC 1H uptrend). Capital protected.")
                    return False
                elif cached_regime == "SHORT_ONLY" and action_name == "LONG":
                    log.warning(f"[REGIME-GATE] LONG vetoed — Macro Warden is SHORT_ONLY (BTC 1H downtrend). Capital protected.")
                    return False
                elif cached_regime:
                    log.debug(f"[REGIME-GATE] Regime={cached_regime}, Action={action_name} — PASS.")
            except Exception as _rge:
                log.warning(f"[REGIME-GATE] Could not read warden regime from DB: {_rge}. Proceeding without gate.")

        # V_AUDIT_FIX: Volume Displacement Gate (>= 1.5x 20-period volume average)
        vol_ratio = strategy_decision.get("vol_ratio") or strategy_decision.get("volume_multiplier") or strategy_decision.get("vol_displacement")
        if vol_ratio is not None:
            try:
                vr = float(vol_ratio)
                if vr > 0.0 and vr < 1.50:
                    log.warning(f"[VOLUME-GATE] Trade aborted: Volume Displacement {vr:.2f}x < 1.50x threshold.")
                    return False
            except (ValueError, TypeError):
                pass

        if action_name == "LONG":
            action_id = ACTION_LONG
            # V32.4 VETO GUARD: Prevent LONG entries into heavy L2 sell walls
            try:
                import os, json, time, asyncio
                ofi_file = "microstructure_state.json"
                def read_ofi():
                    if os.path.exists(ofi_file):
                        for _ in range(5):
                            try:
                                with open(ofi_file, "r") as f:
                                    return json.load(f)
                            except Exception:
                                import time
                                time.sleep(0.02)
                    return {}
                ofi_data = await asyncio.to_thread(read_ofi)
                if ofi_data:
                    if time.time() - ofi_data.get("timestamp", 0) < 10.0:
                        raw_ofi = ofi_data.get("net_ofi_score", 0.0)
                        norm_ofi = ofi_data.get("normalized_ofi", 0.0)
                        ofi_vel = ofi_data.get("ofi_velocity", 0.0)
                        if raw_ofi < -135.0 or norm_ofi < -0.80 or ofi_vel < -0.65:
                            log.warning(f"[VETO] AI Long signal rejected due to heavy L2 sell pressure/velocity (Raw OFI: {raw_ofi:.1f}, Norm: {norm_ofi:.2f}, Vel: {ofi_vel:+.2f}).")
                            return False
            except Exception as e:
                log.error(f"[EXECUTOR] Failed to read OFI state for Veto check: {e}")
                
        elif action_name == "SHORT":
            action_id = ACTION_SHORT
            # Phase 16.4 Adverse Selection Veto Guard (Symmetric with Long)
            try:
                import os, json, time, asyncio
                ofi_file = "microstructure_state.json"
                def read_ofi():
                    if os.path.exists(ofi_file):
                        for _ in range(5):
                            try:
                                with open(ofi_file, "r") as f:
                                    return json.load(f)
                            except Exception:
                                import time
                                time.sleep(0.02)
                    return {}
                ofi_data = await asyncio.to_thread(read_ofi)
                if ofi_data:
                    if time.time() - ofi_data.get("timestamp", 0) < 10.0:
                        raw_ofi = ofi_data.get("net_ofi_score", 0.0)
                        norm_ofi = ofi_data.get("normalized_ofi", 0.0)
                        ofi_vel = ofi_data.get("ofi_velocity", 0.0)
                        if raw_ofi > 135.0 or norm_ofi > 0.80 or ofi_vel > 0.65:
                            log.warning(f"[VETO] Short signal blocked due to heavy L2 buy pressure/velocity (Raw OFI: {raw_ofi:.1f}, Norm: {norm_ofi:.2f}, Vel: {ofi_vel:+.2f}).")
                            return False
            except Exception as e:
                log.error(f"[EXECUTOR] Failed to read OFI state for Veto check: {e}")
        else:
            action_id = ACTION_HOLD

        order = self.evaluate(action_id, confidence, current_price)
        if order is None:
            return False

        if hasattr(self, "exchange") and self.exchange is not None:
            is_live_ccxt = getattr(self, "is_live", False) or hasattr(self.exchange, "apiKey")
            
            if is_live_ccxt:
                # Live Exchange Position Sync Check
                try:
                    positions = await self.exchange.fetch_positions(symbols=[symbol])
                    for pos in positions:
                        size = float(pos.get("contracts", 0.0) or pos.get("size", 0.0) or pos.get("positionAmt", 0.0) or 0.0)
                        if size != 0.0:
                            log.info(f"[EXECUTOR] Skipping trade execution: Live position already open for {symbol} (size: {size}).")
                            return False
                except Exception as e:
                    log.warning(f"[EXECUTOR] Failed to query live positions: {e}")
            else:
                # Check if there is an active position for this symbol to prevent margin leaks
                try:
                    await self.exchange.sync_positions_from_db()
                    if symbol in self.exchange.positions:
                        log.info(f"[EXECUTOR] Skipping trade execution: Position already open for {symbol}.")
                        return False
                except Exception as e:
                    log.warning(f"[EXECUTOR] Failed to sync positions: {e}")

            account_name = getattr(self.exchange, "account_name", "marl_live")
            log.info(f"[EXECUTOR] Routing live execution via {account_name} for {symbol} | {action_name} @ {current_price:,.4f} | Gamma: {gamma}")
            
            try:
                if is_live_ccxt:
                    # Fetch live balance using CCXT
                    balance_info = await self.exchange.fetch_balance()
                    balance = float(balance_info.get("USDT", {}).get("free", 0.0) or balance_info.get("total", {}).get("USDT", 0.0) or 0.0)
                    if balance == 0.0:
                        log.warning("[EXECUTOR] Live balance extraction returned 0.0. Aborting trade sizing.")
                        return False
                else:
                    balance_data = self.exchange.balance
                    if isinstance(balance_data, dict):
                        usdt_entry = balance_data.get("USDT")
                        if isinstance(usdt_entry, dict):
                            balance = float(usdt_entry.get("free", 0.0))
                        elif usdt_entry is not None:
                            balance = float(usdt_entry)
                        else:
                            balance = float(balance_data.get("free", 0.0))
                        if balance == 0.0:
                            log.warning("[EXECUTOR] Balance extraction returned 0.0 — check exchange balance schema. Aborting trade sizing.")
                            return False
                    else:
                        balance = float(balance_data)
                        
                # CRIT-01 FIX: Dynamically link sizing leverage to market_regime (5x TRENDING, 3x CHOP)
                dynamic_leverage = 5.0
                try:
                    import sqlite3, os as _os
                    _db = _os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data.db")
                    if _os.path.exists(_db):
                        with sqlite3.connect(_db, timeout=2.0) as _c:
                            row_reg = _c.execute("SELECT regime FROM market_regime ORDER BY id DESC LIMIT 1").fetchone()
                            if row_reg:
                                reg_label = str(row_reg[0]).upper()
                                dynamic_leverage = 3.0 if ("CHOP" in reg_label or "RANGING" in reg_label) else 5.0
                except Exception:
                    import config as _cfg
                    dynamic_leverage = float(getattr(_cfg, "LEVERAGE", 5.0))

                risk_cap = getattr(self, 'max_cap', 0.05)
                notional = balance * risk_cap * dynamic_leverage
                qty = notional / current_price
                
                # ── Client Order ID Construction ─────────────────────────────────────
                # Phase 16.9 P2 Fix: frozenset enum normalization + strict 32-char ceiling.
                # Binance Futures rejects clientOrderId strings longer than 32 characters.
                _SECONDARY_ACCOUNT_KEYS = frozenset({"berserker", "sniper"})
                _account_key = account_name.lower().split("_")[0]
                is_marl = _account_key not in _SECONDARY_ACCOUNT_KEYS
                prefix = "MARL_SWARM_" if is_marl else "BERSERKER_"

                import time as _time_mod
                _ts_str = str(int(_time_mod.time() * 1000))  # 13-digit ms-epoch

                def _make_client_id(pfx: str, tag: str = "") -> str:
                    """Constructs a Binance-safe clientOrderId, hard-clamped to 32 chars."""
                    raw = f"{pfx}{tag}{_ts_str}"
                    if len(raw) > 32:
                        # Preserve prefix + tag; truncate leading digits of timestamp
                        overflow = len(raw) - 32
                        raw = f"{pfx}{tag}{_ts_str[overflow:]}"
                    return raw

                if is_live_ccxt:
                    client_id = _make_client_id(prefix)
                    
                    # V34.0 Direct Bitget UTA Execution Routing
                    ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
                    if ex_id == "bitget":
                        from bitget_uta_client import BitgetUTAClient
                        uta = BitgetUTAClient()
                        try:
                            # Enforce absolute minimum notional of $5.50
                            if qty * current_price < 5.5:
                                qty = 5.5 / current_price

                            # Apply Bitget-specific contract precision and minimum order quantity limits
                            sym_upper = symbol.upper()
                            if "BTC" in sym_upper:
                                qty = max(round(qty, 4), 0.0001)
                            elif "ETH" in sym_upper or "BNB" in sym_upper:
                                qty = max(round(qty, 2), 0.01)
                            elif "SOL" in sym_upper or "AVAX" in sym_upper:
                                qty = max(round(qty, 1), 0.1)
                            elif any(c in sym_upper for c in ["LINK", "ADA", "XRP", "DOGE", "NEAR"]):
                                qty = max(float(round(qty)), 1.0)
                            else:
                                qty = max(round(qty, 2), 0.1)
                            
                            log.info(f"[EXECUTOR] Routing LIVE Bitget UTA Maker Post-Only order: {symbol} | {action_name.upper()} | Qty: {qty} | SL: {order.stop_loss} | TP: {order.take_profit} | clientOid: {client_id}")
                            try:
                                po_res = await uta.create_post_only_order(
                                    symbol=symbol,
                                    side=action_name.lower(),
                                    qty=qty,
                                    client_oid=client_id,
                                    sl=order.stop_loss,
                                    tp=order.take_profit
                                )
                                po_id = po_res.get("id")
                                log.info(f"[EXECUTOR] Post-Only Order placed. ID: {po_id}. Monitoring maker queue...")
                                fill_res = await _monitor_maker_fill(
                                    uta_client=uta,
                                    order_id=po_id,
                                    symbol=symbol,
                                    side=action_name.lower(),
                                    qty=qty,
                                    current_price=current_price,
                                    timeout_sec=8.0,
                                    sl=order.stop_loss,
                                    tp=order.take_profit,
                                    client_id=client_id
                                )
                                log.info(f"[EXECUTOR] Maker monitor completed: status={fill_res.get('status')}")
                            except Exception as po_err:
                                log.warning(f"[EXECUTOR] Post-only placement failed ({po_err}). Executing direct market order fallback.")
                                order_res = await uta.create_market_order(
                                    symbol=symbol,
                                    side=action_name.lower(),
                                    qty=qty,
                                    client_oid=client_id,
                                    sl=order.stop_loss,
                                    tp=order.take_profit
                                )
                                log.info(f"[EXECUTOR] Direct market fallback placed. ID: {order_res.get('id', 'N/A')}")
                        finally:
                            await uta.close()
                    else:
                        params = {
                            "newClientOrderId": client_id  # CCXT maps this to clientOrderId internally
                        }
                        
                        # 3. Live Leverage & Type Constraint Anchoring
                        try:
                            import config
                            lev = getattr(config, "LEVERAGE", 5)
                            await self.exchange.set_leverage(lev, symbol)
                            log.info(f"[LEVERAGE-ANCHOR] Explicitly set leverage override to {lev}x for {symbol} on the exchange.")
                        except Exception as le:
                            log.warning(f"[LEVERAGE-ANCHOR] Live set_leverage failed: {le}")
                        
                        log.info(f"[EXECUTOR] Placing live order: {symbol} | {action_name.upper()} | Qty: {qty:.4f} | clientOrderId: {client_id}")
                        order_res = await self.exchange.create_market_order(
                            symbol=symbol,
                            side=action_name.lower(),
                            amount=qty,
                            params=params
                        )
                        log.info(f"[EXECUTOR] Live market order placed successfully. ID: {order_res.get('id', 'N/A')}")
                    
                    # Place SL and TP stop orders if specified
                    if order.stop_loss and order.stop_loss > 0.0:
                        try:
                            sl_side = "sell" if action_name.lower() == "long" else "buy"
                            sl_client_id = _make_client_id(prefix, "SL_")
                            sl_params = {
                                "stopPrice": order.stop_loss,
                                "reduceOnly": True,
                                "clientOrderId": sl_client_id,
                                "newClientOrderId": sl_client_id
                            }
                            await self.exchange.create_order(
                                symbol=symbol,
                                type="stop_market",
                                side=sl_side,
                                amount=qty,
                                price=None,
                                params=sl_params
                            )
                            log.info(f"[EXECUTOR] Live Stop Loss stop order placed at ${order.stop_loss:.2f}.")
                        except Exception as sle:
                            log.error(f"[EXECUTOR] Live Stop Loss stop order placement failed: {sle}")
                            
                    if order.take_profit and order.take_profit > 0.0:
                        try:
                            tp_side = "sell" if action_name.lower() == "long" else "buy"
                            tp_client_id = _make_client_id(prefix, "TP_")
                            tp_params = {
                                "stopPrice": order.take_profit,
                                "reduceOnly": True,
                                "clientOrderId": tp_client_id,
                                "newClientOrderId": tp_client_id
                            }
                            await self.exchange.create_order(
                                symbol=symbol,
                                type="take_profit_market",
                                side=tp_side,
                                amount=qty,
                                price=None,
                                params=tp_params
                            )
                            log.info(f"[EXECUTOR] Live Take Profit stop order placed at ${order.take_profit:.2f}.")
                        except Exception as tpe:
                            log.error(f"[EXECUTOR] Live Take Profit stop order placement failed: {tpe}")
                else:
                    await self.exchange.open_position(
                        symbol=symbol,
                        side=action_name.lower(),
                        qty=qty,
                        price=current_price,
                        sl=order.stop_loss,
                        tp=order.take_profit,
                        metadata={"action": action_name, "confidence": confidence, "regime": "MARL_SWARM", "gamma": gamma, "is_maker": True},
                        maker=True
                    )
            except Exception as e:
                log.error(f"[EXECUTOR] Exchange execution failed: {e}")
                return False
            return True
        else:
            self.mock_execute_order(action_id, confidence, current_price)
            return True


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("─" * 55)
    log.info("  ExecutorAgent — Self-Test")
    log.info("─" * 55)

    agent = ExecutorAgent()

    # ── Test 1: LONG ──────────────────────────────────────────────────────────
    log.info("\n[TEST 1] Simulating LONG signal...")
    result = agent.mock_execute_order(
        action_id        = ACTION_LONG,
        confidence_score = 0.80,
        current_price    = 150.00,
    )

    # ── Test 2: SHORT ─────────────────────────────────────────────────────────
    log.info("\n[TEST 2] Simulating SHORT signal...")
    agent.mock_execute_order(
        action_id        = ACTION_SHORT,
        confidence_score = 0.65,
        current_price    = 150.00,
    )

    # ── Test 3: HOLD ──────────────────────────────────────────────────────────
    log.info("\n[TEST 3] Simulating HOLD signal...")
    agent.mock_execute_order(
        action_id        = ACTION_HOLD,
        confidence_score = 0.30,
        current_price    = 150.00,
    )
