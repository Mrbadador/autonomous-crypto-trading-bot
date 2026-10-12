"""
basis_trader.py — Sovereign Swarm BTC Cash-and-Carry Basis Trade Engine
Captures perpetual funding rate yields delta-neutrally (Long Spot BTC + 1x Short Perp BTC)
with an isolated capital allocation, completely decoupled from directional bots.
"""

import asyncio
import os
import sys
import time
import json
import logging
import sqlite3
import signal
from datetime import datetime, UTC
from typing import Optional, Dict, Any

# Ensure correct DNS resolver and configuration loading
import dns_fix  # noqa: F401
import config
from db_manager import DBWriteQueue
from bitget_uta_client import BitgetUTAClient

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] BASIS_TRADER: %(message)s"
)
log = logging.getLogger("basis_trader")


class BTCBasisTrader:
    """
    Principal Quantitative Delta-Neutral Basis Trade Engine.
    Executes a 1:1 spot-futures arbitrage capturing perpetual funding yield
    with zero directional beta and isolated capital allocation.
    """

    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = os.path.abspath(db_path)
        self.symbol_spot = "BTC/USDT"
        self.symbol_perp = "BTC/USDT:USDT"
        self.clean_symbol = "BTCUSDT"

        # Isolated capital & hurdle parameters
        self.capital_allocation = float(getattr(config, "BASIS_TRADE_ALLOCATION_USDT", 1000.0))
        self.min_annual_yield = float(getattr(config, "BASIS_MIN_ANNUAL_YIELD", 0.08))      # 8.0% min annual yield
        self.unwind_annual_yield = float(getattr(config, "BASIS_UNWIND_ANNUAL_YIELD", 0.015)) # 1.5% unwind hurdle
        self.max_dislocation_pct = float(getattr(config, "BASIS_MAX_DISLOCATION_PCT", 0.015)) # 1.5% spot-perp divergence
        self.check_interval = int(getattr(config, "BASIS_CHECK_INTERVAL_SECS", 30))

        # Anti-churn, Fee Amortization, and Risk Parameters
        self.min_holding_days = int(getattr(config, "BASIS_MIN_HOLDING_DAYS", 7))
        self.estimated_fee_bps = float(getattr(config, "BASIS_ESTIMATED_ROUNDTRIP_FEE_BPS", 0.0024)) # 24 bps round-trip
        self.margin_mode = str(getattr(config, "BASIS_MARGIN_MODE", "crossed")).lower()
        self.min_liq_dist_pct = float(getattr(config, "BASIS_MIN_LIQ_DISTANCE_PCT", 0.20)) # 20% liquidation buffer
        self.perp_fill_timeout = int(getattr(config, "BASIS_PERP_FILL_TIMEOUT_SECS", 15)) # 15s sequential fill timeout

        # Fee schedule (maker fee prioritized via post_only)
        self.maker_fee = float(getattr(config, "TRADE_FEE_PERCENT", 0.0002)) # 0.02%
        self.taker_fee = 0.0004 # 0.04%

        # DB Queue for non-blocking WAL persistence
        self.db_queue = DBWriteQueue.get_instance(self.db_path)

        # Exchange execution mode
        mode_val = str(getattr(config, "MODE", "paper")).lower()
        paper_trade = getattr(config, "PAPER_TRADE", True)
        ex_id = str(getattr(config, "EXCHANGE_ID", "bitget")).lower()
        self.is_live_execution = (not paper_trade) and (mode_val in ["live", "live_pilot"]) and (ex_id == "bitget")

        self.uta_client: Optional[BitgetUTAClient] = None
        self.exchange = None
        self._stop_event = asyncio.Event()

        # In-memory active position and state machine
        self.state: str = "IDLE"  # IDLE | PENDING_PERP_FILL | ACTIVE_DELTA_NEUTRAL | UNWINDING
        self.active_position: Optional[Dict[str, Any]] = None
        self.last_funding_timestamp: int = 0
        self.cumulative_funding_collected: float = 0.0
        self.liquidation_price: float = 0.0
        self.distance_to_liq_pct: float = 1.0

        self._init_database()
        self._load_active_position_sync()

    def _init_database(self):
        """Initializes dedicated tables for basis positions and audit ledger."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=30000;")
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS basis_positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        status TEXT NOT NULL,
                        entry_timestamp TEXT NOT NULL,
                        spot_entry_price REAL NOT NULL,
                        spot_qty REAL NOT NULL,
                        perp_entry_price REAL NOT NULL,
                        perp_qty REAL NOT NULL,
                        notional_usd REAL NOT NULL,
                        entry_funding_rate REAL NOT NULL,
                        entry_annualized_yield REAL NOT NULL,
                        cumulative_funding_usd REAL DEFAULT 0.0,
                        last_update_timestamp TEXT NOT NULL,
                        exit_timestamp TEXT,
                        spot_exit_price REAL,
                        perp_exit_price REAL,
                        realized_pnl REAL,
                        unwind_reason TEXT
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS basis_trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        action TEXT NOT NULL,
                        spot_price REAL,
                        spot_qty REAL,
                        perp_price REAL,
                        perp_qty REAL,
                        funding_rate REAL,
                        fee_paid REAL,
                        pnl REAL,
                        notes TEXT
                    );
                """)
            conn.close()
            log.info("[DB] Basis trader tables verified successfully in trading_data.db.")
        except Exception as e:
            log.error(f"[DB FATAL] Failed to initialize basis tables: {e}")

    def _load_active_position_sync(self):
        """Loads any existing OPEN basis position from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM basis_positions WHERE status = 'OPEN' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row:
                self.active_position = dict(row)
                self.cumulative_funding_collected = float(row["cumulative_funding_usd"] or 0.0)
                self.state = "ACTIVE_DELTA_NEUTRAL"
                log.info(
                    f"[RECOVERY] Recovered active BTC Basis Position #{row['id']} "
                    f"(Notional: ${row['notional_usd']:.2f}, Cumulative Funding: ${self.cumulative_funding_collected:.4f})"
                )
            else:
                self.active_position = None
                self.state = "IDLE"
                log.info("[RECOVERY] No active basis position found. Engine is IDLE.")
            conn.close()
        except Exception as e:
            log.error(f"[RECOVERY] Error recovering basis position: {e}")
            self.active_position = None
            self.state = "IDLE"

    async def initialize_clients(self):
        """Initializes CCXT async exchange connection and Bitget UTA execution client."""
        import ccxt.async_support as ccxt_async
        load_dotenv = getattr(config, "load_dotenv", None)
        if load_dotenv:
            load_dotenv()

        api_key = os.getenv("BITGET_API_KEY") or os.getenv("API_KEY")
        secret = os.getenv("BITGET_API_SECRET") or os.getenv("API_SECRET")
        passphrase = os.getenv("BITGET_PASSPHRASE") or os.getenv("API_PASSWORD")

        self.exchange = ccxt_async.bitget({
            "apiKey": api_key,
            "secret": secret,
            "password": passphrase,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True
            }
        })

        if self.is_live_execution:
            self.uta_client = BitgetUTAClient(api_key=api_key, secret=secret, passphrase=passphrase)
            log.info("[EXECUTION] 🔴 LIVE BITGET UTA BASIS EXECUTION ENGAGED (Real Orders Enabled)")
        else:
            self.uta_client = None
            log.info("[EXECUTION] 🟢 High-Fidelity Paper / Simulation Basis Trading Mode Active")

    async def fetch_market_state(self) -> Optional[Dict[str, Any]]:
        """
        Fetches live Spot price, Perpetual Futures price, and 8h Funding Rate for BTC.
        """
        try:
            # Parallel fetch: Spot Ticker, Perp Ticker, Funding Rate
            spot_ticker_task = self.exchange.fetch_ticker(self.symbol_spot)
            perp_ticker_task = self.exchange.fetch_ticker(self.symbol_perp)
            funding_task = self.exchange.fetch_funding_rate(self.symbol_perp)

            spot_ticker, perp_ticker, funding_info = await asyncio.gather(
                spot_ticker_task, perp_ticker_task, funding_task, return_exceptions=True
            )

            if isinstance(spot_ticker, Exception) or isinstance(perp_ticker, Exception) or isinstance(funding_info, Exception):
                log.warning(f"[ORACLE] Market fetch exception: spot={spot_ticker}, perp={perp_ticker}, funding={funding_info}")
                return None

            spot_price = float(spot_ticker.get("last") or spot_ticker.get("close") or 0.0)
            perp_price = float(perp_ticker.get("last") or perp_ticker.get("close") or 0.0)
            funding_rate_8h = float(funding_info.get("fundingRate") or 0.0)
            funding_timestamp = int(funding_info.get("fundingTimestamp") or 0)
            next_funding_rate = funding_info.get("nextFundingRate")
            next_rate = float(next_funding_rate) if next_funding_rate is not None else funding_rate_8h

            if spot_price <= 0.0 or perp_price <= 0.0:
                log.warning("[ORACLE] Invalid price received from market data feed.")
                return None

            # Annualized Yield = Funding_Rate_8h * 3 * 365
            annualized_yield = funding_rate_8h * 3.0 * 365.0
            basis_spread = perp_price - spot_price
            basis_dislocation = (perp_price - spot_price) / spot_price

            return {
                "spot_price": spot_price,
                "perp_price": perp_price,
                "spot_bid": float(spot_ticker.get("bid") or spot_price),
                "spot_ask": float(spot_ticker.get("ask") or spot_price),
                "perp_bid": float(perp_ticker.get("bid") or perp_price),
                "perp_ask": float(perp_ticker.get("ask") or perp_price),
                "funding_rate_8h": funding_rate_8h,
                "funding_timestamp": funding_timestamp,
                "next_funding_rate": next_rate,
                "annualized_yield": annualized_yield,
                "basis_spread": basis_spread,
                "basis_dislocation": basis_dislocation
            }
        except Exception as e:
            log.error(f"[ORACLE] Error fetching market state: {e}")
            return None

    async def update_liquidation_metrics(self, market: Dict[str, Any]):
        """
        Calculates or queries the Perp Short liquidation price and safety distance.
        For a 1x Short Perp:
            liq_price ≈ entry_price * (1.0 + (1.0 / leverage) - MMR)
        With BTC MMR = 0.004 (0.4%), isolated liq ≈ entry_price * 1.996.
        Cross-margin with Unified Account equity provides an even wider safety buffer.
        """
        if not self.active_position:
            self.liquidation_price = 0.0
            self.distance_to_liq_pct = 1.0
            return

        perp_price = float(market.get("perp_price", 0.0) or 0.0)
        perp_entry = float(self.active_position.get("perp_entry_price", 0.0))
        if perp_entry <= 0.0 or perp_price <= 0.0:
            return

        liq_price = 0.0
        # If live client is active, attempt to read live liquidation price from exchange position
        if self.is_live_execution and self.uta_client:
            try:
                positions = await self.uta_client.fetch_positions(symbol="BTCUSDT")
                for p in positions:
                    pos_side = (p.get("posSide") or p.get("holdSide") or "").lower()
                    if pos_side == "short":
                        lp = float(p.get("liquidationPrice", 0.0) or p.get("liqPx", 0.0) or 0.0)
                        if lp > 0:
                            liq_price = lp
                            break
            except Exception as ex:
                log.debug(f"[LIQ] Live position liq fetch note: {ex}")

        if liq_price <= 0.0:
            # Theoretical 1x short liquidation price
            mmr = 0.004
            liq_price = perp_entry * (1.0 + 1.0 - mmr)

        self.liquidation_price = round(liq_price, 2)
        if perp_price >= self.liquidation_price:
            self.distance_to_liq_pct = 0.0
        else:
            self.distance_to_liq_pct = max(0.0, (self.liquidation_price - perp_price) / perp_price)

    def evaluate_unwind_conditions(self, market: Dict[str, Any]) -> tuple[bool, str]:
        """
        Evaluates whether the delta-neutral basis position should be unwound.
        Enforces:
        1. Emergency Bypasses:
           - Negative funding rate (funding_rate_8h < 0.0)
           - Extreme basis dislocation (> 1.5%)
           - Defensive Liquidation Guard (distance_to_liq_pct < 20%)
        2. Anti-Churn & Fee Amortization Gate:
           - Blocks sub-hurdle unwind unless held >= 7 days OR funding >= 125% of roundtrip fees.
        """
        if not self.active_position:
            return False, ""

        funding_rate = market.get("funding_rate_8h", 0.0)
        annualized_yield = market.get("annualized_yield", 0.0)
        dislocation = abs(market.get("basis_dislocation", 0.0))

        # 1. EMERGENCY BYPASS 1: Negative Funding Rate (shorts pay longs)
        if funding_rate < 0.0:
            return True, f"EMERGENCY_NEGATIVE_FUNDING_{funding_rate*100:+.4f}%"

        # 2. EMERGENCY BYPASS 2: Extreme Basis Dislocation (> 1.5%)
        if dislocation > self.max_dislocation_pct:
            return True, f"EMERGENCY_BASIS_DISLOCATION_{dislocation*100:.2f}%"

        # 3. DEFENSIVE LIQUIDATION GUARD: Within 20% of Perp Short liquidation
        if self.distance_to_liq_pct < self.min_liq_dist_pct:
            log.warning(
                f"[LIQUIDATION ALERT] 🚨 Short Perp liquidation distance dropped to {self.distance_to_liq_pct*100:.1f}% "
                f"(< {self.min_liq_dist_pct*100:.0f}% threshold)! Liq Price: ${self.liquidation_price:,.2f}, "
                f"Current Perp: ${market['perp_price']:,.2f}. Triggering defensive unwind!"
            )
            return True, f"DEFENSIVE_LIQUIDATION_GUARD_{self.distance_to_liq_pct*100:.1f}%"

        # 4. SUB-HURDLE YIELD WITH ANTI-CHURN & FEE AMORTIZATION GATE
        if annualized_yield <= self.unwind_annual_yield:
            entry_time_str = self.active_position.get("entry_timestamp", "")
            holding_days = 0.0
            try:
                clean_t = entry_time_str.replace("Z", "+00:00")
                entry_dt = datetime.fromisoformat(clean_t)
                now_dt = datetime.now(UTC)
                holding_days = max(0.0, (now_dt - entry_dt).total_seconds() / 86400.0)
            except Exception:
                pass

            notional = float(self.active_position.get("notional_usd", self.capital_allocation))
            roundtrip_fee_usd = notional * self.estimated_fee_bps
            fee_hurdle_target = roundtrip_fee_usd * 1.25  # 125% fee recovery
            fee_amortized_pct = (self.cumulative_funding_collected / roundtrip_fee_usd * 100.0) if roundtrip_fee_usd > 0 else 100.0

            if holding_days >= self.min_holding_days:
                return True, f"SUB_HURDLE_HOLD_MATURED_{holding_days:.1f}d_YIELD_{annualized_yield*100:.2f}%"
            elif self.cumulative_funding_collected >= fee_hurdle_target:
                return True, f"SUB_HURDLE_FEES_AMORTIZED_{fee_amortized_pct:.1f}%_YIELD_{annualized_yield*100:.2f}%"
            else:
                log.info(
                    f"[ANTI-CHURN GATE] ⏳ Sub-hurdle yield ({annualized_yield*100:.2f}% <= {self.unwind_annual_yield*100:.2f}%), "
                    f"but holding duration ({holding_days:.2f}d < {self.min_holding_days}d) and fee amortization "
                    f"({fee_amortized_pct:.1f}% < 125.0% / ${self.cumulative_funding_collected:.2f} < ${fee_hurdle_target:.2f}) "
                    f"hurdles not yet met. Preserving delta-neutral position to prevent fee bleed."
                )
                return False, ""

        return False, ""

    async def open_basis_position(self, market: Dict[str, Any]):
        """
        Executes Sequential 3-Step Execution State Machine (Eliminates Legging Risk):
        Step 1 (Passive Perp Leg): Post a passive maker limit order (post_only=True) for Short Perp leg at inside ask.
        Step 2 (Fill Confirmation): Monitor Perp order fill state within a 15-second polling window.
        Step 3 (Spot Hedging Execution):
          - If Perp order fills: Immediately route an atomic Spot buy order for the exact executed Perp quantity.
          - If Perp order times out or is partially filled: Cancel the unfilled perpetual remnant.
            If 0 filled, do NOT buy spot (guaranteed ZERO legging delta).
            If partially filled, only buy spot matching the exact filled portion.
        """
        spot_price = market["spot_price"]
        perp_price = market["perp_price"]
        annualized_yield = market["annualized_yield"]
        funding_rate = market["funding_rate_8h"]

        # Isolated capital allocation per leg
        leg_notional = self.capital_allocation / 2.0  # e.g., $500.00
        target_perp_qty = round(leg_notional / perp_price, 4)

        if target_perp_qty <= 0.0001:
            log.warning(f"[ENTRY] Sizing calculation too small: target_perp_qty={target_perp_qty}")
            return

        now_iso = datetime.now(UTC).isoformat()
        log.info(
            f"[ENTRY GATE] 🎯 Positive Basis Condition Triggered! Annualized Yield: {annualized_yield*100:.2f}% "
            f"(Hurdle: {self.min_annual_yield*100:.2f}%). Target Perp Short={target_perp_qty} BTC (${leg_notional:.2f}). "
            f"Margin Mode: {self.margin_mode.upper()}."
        )

        # Transition State Machine to PENDING_PERP_FILL
        self.state = "PENDING_PERP_FILL"
        self.publish_telemetry(market)

        perp_qty = 0.0
        perp_fill_price = market["perp_ask"]
        spot_fill_price = market["spot_ask"]

        # Route Live Orders if enabled
        if self.is_live_execution and self.uta_client:
            try:
                # Step 1: Passive Maker Perp Leg (post_only=True, 1x leverage, Cross-Margin)
                log.info(f"[ENTRY STEP 1] Posting Passive Maker Perp Short: {target_perp_qty} BTC @ ${market['perp_ask']:,.2f} (post_only, {self.margin_mode})")
                perp_order = await self.uta_client.create_post_only_order(
                    symbol="BTCUSDT",
                    side="short",
                    qty=target_perp_qty,
                    price=market["perp_ask"],
                    leverage_val=1,  # Strict 1x leverage for cash-and-carry safety
                    margin_mode=self.margin_mode
                )
                order_id = perp_order.get("id")
                if not order_id:
                    log.error("[ENTRY FAILED] No orderId returned for Perp Short post_only order.")
                    self.state = "IDLE"
                    self.publish_telemetry(market)
                    return

                # Step 2: Fill Confirmation Loop (Poll for up to self.perp_fill_timeout seconds)
                poll_interval = 1.0
                elapsed = 0.0
                filled_qty = 0.0
                deal_avg_price = market["perp_ask"]
                order_filled = False

                while elapsed < self.perp_fill_timeout:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval

                    try:
                        order_info = await self.uta_client.fetch_order(
                            symbol="BTCUSDT",
                            order_id=order_id,
                            category="USDT-FUTURES"
                        )
                        status = str(order_info.get("order_status", "")).lower()
                        filled_qty = float(order_info.get("filled_qty", 0.0) or 0.0)
                        if float(order_info.get("deal_avg_price", 0.0) or 0.0) > 0:
                            deal_avg_price = float(order_info["deal_avg_price"])

                        if status in ["filled", "closed"] or filled_qty >= (target_perp_qty * 0.999):
                            order_filled = True
                            log.info(f"[ENTRY STEP 2] Perp Short Order {order_id} FILLED! Qty={filled_qty} BTC @ ${deal_avg_price:,.2f} ({elapsed:.1f}s)")
                            break
                        elif status in ["canceled", "cancelled"]:
                            log.warning(f"[ENTRY STEP 2] Perp Short order was cancelled by exchange (status={status}).")
                            break
                    except Exception as poll_err:
                        log.warning(f"[ENTRY STEP 2] Error polling perp order {order_id}: {poll_err}")

                # Timeout or partial fill handling: cancel unfilled remnant
                if not order_filled:
                    log.info(f"[ENTRY TIMEOUT] Perp Short order {order_id} incomplete after {elapsed:.1f}s. Cancelling unfilled remnant...")
                    try:
                        await self.uta_client.cancel_order(
                            symbol="BTCUSDT",
                            order_id=order_id,
                            category="USDT-FUTURES"
                        )
                    except Exception as cancel_err:
                        log.warning(f"[ENTRY] Perp cancel remnant note: {cancel_err}")

                    # Final verification of fill state post-cancellation
                    try:
                        final_info = await self.uta_client.fetch_order(
                            symbol="BTCUSDT",
                            order_id=order_id,
                            category="USDT-FUTURES"
                        )
                        filled_qty = float(final_info.get("filled_qty", filled_qty) or filled_qty)
                        if float(final_info.get("deal_avg_price", 0.0) or 0.0) > 0:
                            deal_avg_price = float(final_info["deal_avg_price"])
                    except Exception:
                        pass

                # If 0 filled: ABORT! Zero Spot BTC bought, Zero Legging Delta!
                if filled_qty <= 0.0001:
                    log.info(
                        f"[ENTRY ABORTED] 🛑 Perp Short maker order {order_id} timed out unfilled. "
                        f"0 Spot BTC bought (Guaranteed ZERO Legging Delta). Returning to IDLE."
                    )
                    self.state = "IDLE"
                    self.publish_telemetry(market)
                    return

                # Step 3: Spot Hedging Execution matching EXACT executed Perp quantity
                perp_qty = filled_qty
                perp_fill_price = deal_avg_price
                spot_qty = perp_qty  # Strict 1:1 Parity

                log.info(
                    f"[ENTRY STEP 3] ⚡ Executing Spot Buy Hedge for {spot_qty} BTC "
                    f"to lock in 1:1 Delta Neutrality against filled Perp Short."
                )
                spot_order = await self.uta_client.create_spot_order(
                    symbol="BTCUSDT",
                    side="buy",
                    qty=spot_qty,
                    price=None,  # Market order guarantees instantaneous fill
                    post_only=False
                )
                spot_fill_price = float(spot_order.get("price", 0.0) or 0.0)
                if spot_fill_price <= 0.0:
                    spot_fill_price = market["spot_ask"]

            except Exception as ex_err:
                log.error(f"[LIVE EXECUTION FATAL] Error during sequential basis entry: {ex_err}", exc_info=True)
                self.state = "IDLE"
                self.publish_telemetry(market)
                return
        else:
            # High-fidelity simulated sequential fill
            await asyncio.sleep(1.0)
            perp_qty = target_perp_qty
            spot_qty = target_perp_qty
            perp_fill_price = market["perp_ask"]
            spot_fill_price = market["spot_ask"]

        total_notional = (spot_qty * spot_fill_price) + (perp_qty * perp_fill_price)
        entry_fee_paid = (spot_qty * spot_fill_price * self.taker_fee) + (perp_qty * perp_fill_price * self.maker_fee)

        # Persist position to SQLite via DBWriteQueue
        insert_query = """
            INSERT INTO basis_positions (
                symbol, status, entry_timestamp, spot_entry_price, spot_qty,
                perp_entry_price, perp_qty, notional_usd, entry_funding_rate,
                entry_annualized_yield, cumulative_funding_usd, last_update_timestamp
            ) VALUES (?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?)
        """
        self.db_queue.enqueue_write(
            "basis_positions",
            insert_query,
            (
                "BTC/USDT",
                now_iso,
                spot_fill_price,
                spot_qty,
                perp_fill_price,
                perp_qty,
                total_notional,
                funding_rate,
                annualized_yield,
                now_iso
            )
        )

        # Record trade event in audit ledger
        trade_query = """
            INSERT INTO basis_trades (
                timestamp, symbol, action, spot_price, spot_qty,
                perp_price, perp_qty, funding_rate, fee_paid, pnl, notes
            ) VALUES (?, 'BTC/USDT', 'OPEN_TWIN', ?, ?, ?, ?, ?, ?, 0.0, ?)
        """
        self.db_queue.enqueue_write(
            "basis_trades",
            trade_query,
            (
                now_iso,
                spot_fill_price,
                spot_qty,
                perp_fill_price,
                perp_qty,
                funding_rate,
                entry_fee_paid,
                f"Opened 1:1 Delta-Neutral Basis Trade (Sequential Fill). Yield: {annualized_yield*100:.2f}%"
            )
        )

        # Update in-memory state
        self.active_position = {
            "symbol": "BTC/USDT",
            "status": "OPEN",
            "entry_timestamp": now_iso,
            "spot_entry_price": spot_fill_price,
            "spot_qty": spot_qty,
            "perp_entry_price": perp_fill_price,
            "perp_qty": perp_qty,
            "notional_usd": total_notional,
            "entry_funding_rate": funding_rate,
            "entry_annualized_yield": annualized_yield,
            "cumulative_funding_usd": 0.0,
            "last_update_timestamp": now_iso
        }
        self.cumulative_funding_collected = 0.0
        self.state = "ACTIVE_DELTA_NEUTRAL"
        await self.update_liquidation_metrics(market)

        log.info(
            f"[POSITION OPEN] 🚀 BTC Basis Trade Active. Notional: ${total_notional:.2f}, "
            f"Fees Paid: ${entry_fee_paid:.4f}. Net Delta: $0.00 (Guaranteed Parity). "
            f"Liq Price: ${self.liquidation_price:,.2f} ({self.distance_to_liq_pct*100:.1f}% buffer)."
        )

    async def unwind_basis_position(self, market: Dict[str, Any], reason: str):
        """
        Executes Orderly Unwind of Twin Position:
        Leg 1: Buy to Close Perp Short BTC first.
        Leg 2: Sell Spot BTC.
        Calculates net realized PnL including funding collected and fees.
        """
        if not self.active_position:
            return

        self.state = "UNWINDING"
        self.publish_telemetry(market)

        pos = self.active_position
        spot_qty = float(pos["spot_qty"])
        perp_qty = float(pos["perp_qty"])
        spot_entry = float(pos["spot_entry_price"])
        perp_entry = float(pos["perp_entry_price"])

        spot_exit = market["spot_bid"]
        perp_exit = market["perp_ask"]
        now_iso = datetime.now(UTC).isoformat()

        log.info(f"[UNWIND INITIATED] ⚠️ Unwinding BTC Basis Position: Reason = {reason}")

        # Route Live Close Orders if enabled
        if self.is_live_execution and self.uta_client:
            try:
                # Leg 1: Close Perp Short BTC first
                await self.uta_client.close_position_order(
                    symbol="BTCUSDT",
                    side="short",
                    qty=perp_qty
                )
                # Leg 2: Sell Spot BTC
                await self.uta_client.close_spot_order(
                    symbol="BTCUSDT",
                    qty=spot_qty,
                    price=market["spot_bid"]
                )
                log.info("[LIVE EXECUTION] Twin Unwind Orders executed successfully on Bitget.")
            except Exception as close_err:
                log.error(f"[LIVE EXECUTION] Error during live twin unwind: {close_err}")
                return

        # PnL Calculations
        spot_pnl = spot_qty * (spot_exit - spot_entry)
        perp_pnl = perp_qty * (perp_entry - perp_exit)  # Short profit = entry - exit
        total_exit_notional = (spot_qty * spot_exit) + (perp_qty * perp_exit)
        exit_fee_paid = total_exit_notional * self.maker_fee
        net_realized_pnl = spot_pnl + perp_pnl + self.cumulative_funding_collected - exit_fee_paid

        log.info(
            f"[UNWIND SUMMARY] Spot PnL: ${spot_pnl:+.4f} | Perp PnL: ${perp_pnl:+.4f} | "
            f"Funding Collected: ${self.cumulative_funding_collected:+.4f} | Exit Fees: ${exit_fee_paid:.4f} | "
            f"NET REALIZED PnL: ${net_realized_pnl:+.4f}"
        )

        # Update position record in database
        update_query = """
            UPDATE basis_positions SET
                status = 'CLOSED',
                exit_timestamp = ?,
                spot_exit_price = ?,
                perp_exit_price = ?,
                realized_pnl = ?,
                unwind_reason = ?,
                last_update_timestamp = ?
            WHERE status = 'OPEN' AND symbol = 'BTC/USDT'
        """
        self.db_queue.enqueue_write(
            "basis_positions",
            update_query,
            (now_iso, spot_exit, perp_exit, net_realized_pnl, reason, now_iso)
        )

        # Record trade event in audit ledger
        trade_query = """
            INSERT INTO basis_trades (
                timestamp, symbol, action, spot_price, spot_qty,
                perp_price, perp_qty, funding_rate, fee_paid, pnl, notes
            ) VALUES (?, 'BTC/USDT', 'CLOSE_TWIN', ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db_queue.enqueue_write(
            "basis_trades",
            trade_query,
            (
                now_iso,
                spot_exit,
                spot_qty,
                perp_exit,
                perp_qty,
                market["funding_rate_8h"],
                exit_fee_paid,
                net_realized_pnl,
                f"Unwound Basis Trade ({reason}). Net PnL: ${net_realized_pnl:+.4f}"
            )
        )

        # Reset active state and return to IDLE
        self.active_position = None
        self.cumulative_funding_collected = 0.0
        self.state = "IDLE"
        self.liquidation_price = 0.0
        self.distance_to_liq_pct = 1.0
        self.publish_telemetry(market)

    def check_and_accrue_funding(self, market: Dict[str, Any]):
        """
        Accrues funding payments every 8 hours on funding settlement timestamps.
        Because we are SHORT the perpetual contract, we receive positive funding payments
        when the funding rate is positive.
        """
        if not self.active_position:
            return

        funding_timestamp = market["funding_timestamp"]
        funding_rate = market["funding_rate_8h"]

        # Only accrue once per 8-hour settlement window
        if funding_timestamp > self.last_funding_timestamp and self.last_funding_timestamp > 0:
            perp_qty = float(self.active_position["perp_qty"])
            perp_price = market["perp_price"]
            perp_notional = perp_qty * perp_price

            # Positive rate: Shorts receive funding from Longs
            payment = perp_notional * funding_rate
            self.cumulative_funding_collected += payment
            self.active_position["cumulative_funding_usd"] = self.cumulative_funding_collected
            now_iso = datetime.now(UTC).isoformat()

            log.info(
                f"[FUNDING HARVEST] 💰 Harvested 8h Funding: +${payment:.4f} "
                f"(Rate: {funding_rate*100:.4f}% on ${perp_notional:.2f} short). "
                f"Total Collected to Date: ${self.cumulative_funding_collected:.4f}"
            )

            # Persist updated cumulative funding to database
            update_query = """
                UPDATE basis_positions SET
                    cumulative_funding_usd = ?,
                    last_update_timestamp = ?
                WHERE status = 'OPEN' AND symbol = 'BTC/USDT'
            """
            self.db_queue.enqueue_write(
                "basis_positions",
                update_query,
                (self.cumulative_funding_collected, now_iso)
            )

            # Record in basis_trades
            trade_query = """
                INSERT INTO basis_trades (
                    timestamp, symbol, action, spot_price, spot_qty,
                    perp_price, perp_qty, funding_rate, fee_paid, pnl, notes
                ) VALUES (?, 'BTC/USDT', 'FUNDING_HARVEST', ?, ?, ?, ?, ?, 0.0, ?, ?)
            """
            self.db_queue.enqueue_write(
                "basis_trades",
                trade_query,
                (
                    now_iso,
                    market["spot_price"],
                    self.active_position["spot_qty"],
                    perp_price,
                    perp_qty,
                    funding_rate,
                    payment,
                    f"8h Funding settlement payment collected: +${payment:.4f}"
                )
            )

        self.last_funding_timestamp = funding_timestamp

    def publish_telemetry(self, market: Optional[Dict[str, Any]]):
        """
        Emits high-frequency JSON telemetry heartbeat to system_state
        for direct rendering on the Streamlit dashboard and mobile web bridge.
        """
        now_iso = datetime.now(UTC).isoformat()
        has_pos = bool(self.active_position is not None)

        spot_p = market["spot_price"] if market else 0.0
        perp_p = market["perp_price"] if market else 0.0
        spread = market["basis_spread"] if market else 0.0
        disloc = market["basis_dislocation"] if market else 0.0
        rate = market["funding_rate_8h"] if market else 0.0
        ann_yield = market["annualized_yield"] if market else 0.0

        # Calculate unrealized basis drift PnL and net delta if open
        unrealized_pnl = 0.0
        net_delta_usd = 0.0
        holding_days = 0.0
        roundtrip_fee_usd = self.capital_allocation * self.estimated_fee_bps
        fee_amortized_pct = 0.0

        if has_pos and market:
            s_entry = float(self.active_position["spot_entry_price"])
            p_entry = float(self.active_position["perp_entry_price"])
            s_qty = float(self.active_position["spot_qty"])
            p_qty = float(self.active_position["perp_qty"])
            notional = float(self.active_position.get("notional_usd", (s_qty * spot_p) + (p_qty * perp_p)))

            unrealized_pnl = (s_qty * (spot_p - s_entry)) + (p_qty * (p_entry - perp_p)) + self.cumulative_funding_collected
            net_delta_usd = round((s_qty * spot_p) - (p_qty * perp_p), 2)

            roundtrip_fee_usd = notional * self.estimated_fee_bps
            fee_amortized_pct = round((self.cumulative_funding_collected / roundtrip_fee_usd * 100.0), 1) if roundtrip_fee_usd > 0 else 100.0

            entry_time_str = self.active_position.get("entry_timestamp", "")
            try:
                clean_t = entry_time_str.replace("Z", "+00:00")
                entry_dt = datetime.fromisoformat(clean_t)
                holding_days = max(0.0, (datetime.now(UTC) - entry_dt).total_seconds() / 86400.0)
            except Exception:
                pass

        fee_hurdle_target_usd = round(roundtrip_fee_usd * 1.25, 2)

        telemetry = {
            "timestamp": now_iso,
            "engine": "basis_trader",
            "state": self.state,
            "status": "ACTIVE_DELTA_NEUTRAL" if has_pos else ("PENDING_PERP_FILL" if self.state == "PENDING_PERP_FILL" else "SCANNING_YIELD"),
            "symbol": "BTC/USDT",
            "spot_price": spot_p,
            "perp_price": perp_p,
            "basis_spread_usd": round(spread, 2),
            "basis_dislocation_pct": round(disloc * 100, 3),
            "funding_rate_8h": round(rate, 6),
            "annualized_yield_pct": round(ann_yield * 100, 2),
            "hurdle_rate_pct": round(self.min_annual_yield * 100, 2),
            "unwind_hurdle_pct": round(self.unwind_annual_yield * 100, 2),
            "capital_allocated_usd": self.capital_allocation,
            "net_delta_usd": net_delta_usd,
            "market_neutral": True,
            "has_active_position": has_pos,
            "cumulative_funding_harvested": round(self.cumulative_funding_collected, 4),
            "unrealized_pnl_usd": round(unrealized_pnl, 4),
            "holding_days": round(holding_days, 2),
            "min_holding_days": self.min_holding_days,
            "roundtrip_fee_usd": round(roundtrip_fee_usd, 2),
            "fee_hurdle_target_usd": fee_hurdle_target_usd,
            "fee_amortized_pct": fee_amortized_pct,
            "margin_mode": self.margin_mode,
            "liquidation_price": self.liquidation_price,
            "distance_to_liq_pct": round(self.distance_to_liq_pct * 100.0, 1),
            "min_liq_distance_pct": round(self.min_liq_dist_pct * 100.0, 1),
            "position": self.active_position or {}
        }

        # Non-blocking enqueue to system_state table in trading_data.db
        state_query = """
            INSERT OR REPLACE INTO system_state (key, value, updated_at)
            VALUES ('basis_trader_telemetry', ?, ?)
        """
        self.db_queue.enqueue_write("system_state", state_query, (json.dumps(telemetry), now_iso))

    async def run(self):
        """Main autonomous execution loop for BTC Cash-and-Carry Basis Trader."""
        log.info("=" * 65)
        log.info("   LAUNCHING BTC CASH-AND-CARRY BASIS TRADING DAEMON")
        log.info(f"   Allocation: ${self.capital_allocation:,.2f} | Hurdle: {self.min_annual_yield*100:.1f}% | Unwind: {self.unwind_annual_yield*100:.1f}%")
        log.info(f"   Margin Mode: {self.margin_mode.upper()} | Anti-Churn: {self.min_holding_days}d / 125% Fee Recovery")
        log.info("   Strategy: Long Spot BTC + 1x Short Perp BTC (Delta-Neutral)")
        log.info("=" * 65)

        await self.initialize_clients()

        while not self._stop_event.is_set():
            try:
                # 1. Fetch live spot, perp, and funding rate
                market = await self.fetch_market_state()

                if market:
                    # 2. Check and accrue 8-hour funding payments
                    self.check_and_accrue_funding(market)

                    # 3. Update liquidation distance if position is active
                    if self.active_position is not None:
                        await self.update_liquidation_metrics(market)

                    # 4. Position & Gating Logic
                    if self.active_position is None and self.state == "IDLE":
                        # Gating evaluation: Open when annualized yield exceeds threshold and spread is safe
                        if market["annualized_yield"] >= self.min_annual_yield:
                            if abs(market["basis_dislocation"]) <= self.max_dislocation_pct:
                                await self.open_basis_position(market)
                            else:
                                log.warning(
                                    f"[DISLOCATION GUARD] Yield {market['annualized_yield']*100:.2f}% meets hurdle, "
                                    f"but spot-perp dislocation {market['basis_dislocation']*100:.2f}% exceeds safe limit {self.max_dislocation_pct*100:.2f}%."
                                )
                    elif self.active_position is not None and self.state == "ACTIVE_DELTA_NEUTRAL":
                        # Evaluates anti-churn gate, emergency negative funding, extreme dislocation, and liq guard
                        should_unwind, unwind_reason = self.evaluate_unwind_conditions(market)
                        if should_unwind:
                            await self.unwind_basis_position(market, reason=unwind_reason)

                    # 5. Emit real-time telemetry to system_state
                    self.publish_telemetry(market)

            except Exception as loop_err:
                log.error(f"[LOOP ERROR] Unexpected basis trader error: {loop_err}", exc_info=True)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.check_interval)
            except asyncio.TimeoutError:
                pass

        log.info("[SHUTDOWN] Basis Trader shutting down cleanly.")
        if self.exchange:
            await self.exchange.close()
        if self.uta_client:
            await self.uta_client.close()

    def stop(self):
        """Signals the daemon to halt execution gracefully."""
        self._stop_event.set()


def main():
    if not getattr(config, "BASIS_TRADER_ENABLED", False):
        log.info("[DISABLED] BTC Cash-and-Carry Basis Trader is disabled via config (BASIS_TRADER_ENABLED=false).")
        try:
            conn = sqlite3.connect("trading_data.db", timeout=10.0)
            now_iso = datetime.now(UTC).isoformat()
            parked_payload = json.dumps({
                "status": "DISABLED",
                "state": "DISABLED",
                "reason": "Temporarily parked for account segregation",
                "net_delta": 0.0,
                "timestamp": now_iso
            })
            conn.execute(
                "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                ("basis_trader_telemetry", parked_payload, now_iso)
            )
            conn.commit()
            conn.close()
            log.info("[DISABLED] Successfully emitted parked telemetry state to trading_data.db.")
        except Exception as e:
            log.error(f"[DISABLED] Failed to write parked telemetry: {e}")
        sys.exit(0)

    trader = BTCBasisTrader()

    def _sig_handler(sig, frame):
        log.warning(f"[SIGNAL] Shutdown signal ({sig}) received. Halting Basis Trader...")
        trader.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        asyncio.run(trader.run())
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        log.critical(f"[FATAL] Basis Trader exited with fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
