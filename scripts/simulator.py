import time
from datetime import datetime, UTC
import logging
import config
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

@dataclass
class SimulatedPosition:
    id: int
    symbol: str
    side: str              # "long" or "short"
    entry_price: float
    qty: float             # Size in base currency
    leverage: float
    timestamp: datetime
    liquidation_price: float
    unr: float = 0.0      # Unrealized PnL
    realized_pnl: float = 0.0
    is_open: bool = True
    
    # Metadata for bot strategy
    sl: float = 0.0
    tp: float = 0.0
    sl_mode: str = "normal"
    tp_mode: str = "swing"
    trail_mode: str = "normal"
    sl_offset: float = 0.0
    regime: str = "UNKNOWN"
    phase: str = "UNKNOWN"
    quality: float = 0.0
    notes: str = ""
    partial_taken: bool = False   # True after the first 50% partial TP fires
    peak_price: float = 0.0       # Highest price (long) or lowest price (short) reached
    mfe: float = 0.0              # Max Favorable Excursion in percent

@dataclass
class SimulatedBalance:
    total: float
    available: float
    used: float = 0.0

class SimulatedExchange:
    def __init__(self, starting_balance: float = 10000.0, maintenance_margin: float = 0.005):
        self.balance = SimulatedBalance(total=starting_balance, available=starting_balance)
        self.positions: Dict[int, SimulatedPosition] = {} # Keyed by ID for multi-position symbol support
        self.maintenance_margin = maintenance_margin
        self.last_prices: Dict[str, float] = {}
        self.next_id = 1

    def update_price(self, symbol: str, price: float):
        self.last_prices[symbol] = price
        self._update_all_positions_for_symbol(symbol, price)

    def open_position(self, symbol: str, side: str, qty: float, price: float, leverage: float, sl: float = 0.0, tp: float = 0.0, **metadata):
        notional = qty * price

        # Deduct trading fee
        fee = notional * config.TRADE_FEE_PERCENT
        if fee > self.balance.available:
            log.warning(f"SIM: Insufficient balance to pay fee {fee:.2f} for {symbol}")
            return None
        self.balance.available -= fee
        self.balance.total -= fee

        margin_required = notional / leverage
        if margin_required > self.balance.available:
            log.warning(f"SIM: Insufficient margin for {symbol}. Required: {margin_required}, Available: {self.balance.available}")
            return None

        liq_price = self._calc_liquidation_price(price, side, leverage)

        pos = SimulatedPosition(
            id=self.next_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            qty=qty,
            leverage=leverage,
            timestamp=datetime.now(UTC),
            liquidation_price=liq_price,
            sl=sl,
            tp=tp,
            peak_price=price,
            **metadata
        )
        self.positions[pos.id] = pos
        self.balance.available -= margin_required
        self.balance.used += margin_required
        self.next_id += 1
        return pos

    def close_position(self, pos_id: int, price: Optional[float] = None):
        if pos_id not in self.positions or not self.positions[pos_id].is_open:
            return None

        pos = self.positions[pos_id]
        price = price if price is not None else self.last_prices.get(pos.symbol, pos.entry_price)

        pnl = self._calc_pnl(pos, price)
        # Deduct trading fee on close
        notional = pos.qty * price
        fee = notional * config.TRADE_FEE_PERCENT
        pnl -= fee

        pos.realized_pnl += pnl
        pos.unr = 0.0
        pos.is_open = False

        notional_entry = pos.qty * pos.entry_price
        margin = notional_entry / pos.leverage

        self.balance.used -= margin
        self.balance.available += (margin + pnl)
        self.balance.total += pnl

        del self.positions[pos_id]
        return pos

    def _update_all_positions_for_symbol(self, symbol: str, price: float):
        for pos in list(self.positions.values()):
            if pos.symbol == symbol and pos.is_open:
                pos.unr = self._calc_pnl(pos, price)

                # Update peak_price and MFE — must be inside the loop
                if pos.side == "long":
                    if price > pos.peak_price:
                        pos.peak_price = price
                        pos.mfe = max(pos.mfe, (price - pos.entry_price) / pos.entry_price * 100)
                else:
                    if price < pos.peak_price or pos.peak_price == 0:
                        pos.peak_price = price
                        pos.mfe = max(pos.mfe, (pos.entry_price - price) / pos.entry_price * 100)

                # Liquidation Check — applies to BOTH sides
                notional = pos.qty * pos.entry_price
                margin = notional / pos.leverage
                equity_on_pos = margin + pos.unr

                if equity_on_pos <= notional * self.maintenance_margin:
                    log.critical(f"SIM: !!! LIQUIDATION !!! {symbol} (ID:{pos.id}) at {price}")
                    self._liquidate_position(pos.id, price)

    def _liquidate_position(self, pos_id: int, price: float):
        pos = self.positions[pos_id]
        pnl = self._calc_pnl(pos, price)
        pos.is_open = False

        notional = pos.qty * pos.entry_price
        margin = notional / pos.leverage
        self.balance.used -= margin
        
        equity_on_pos = margin + pnl
        self.balance.available += max(equity_on_pos, 0)
        self.balance.total += pnl
        
        del self.positions[pos_id]

    def _calc_pnl(self, pos: SimulatedPosition, price: float) -> float:
        direction = 1 if pos.side == "long" else -1
        return (price - pos.entry_price) * pos.qty * direction

    def _calc_pnl_for_qty(self, pos: SimulatedPosition, price: float, qty: float) -> float:
        direction = 1 if pos.side == "long" else -1
        return (price - pos.entry_price) * qty * direction

    def close_partial_position(self, pos_id: int, fraction: float, price: Optional[float] = None) -> Optional[float]:
        if pos_id not in self.positions or not self.positions[pos_id].is_open:
            return None
        if not (0 < fraction < 1):
            log.warning(f"SIM: Invalid fraction {fraction} for partial close of pos {pos_id}")
            return None

        pos = self.positions[pos_id]
        price = price if price is not None else self.last_prices.get(pos.symbol, pos.entry_price)
        close_qty = pos.qty * fraction
        pnl = self._calc_pnl_for_qty(pos, price, close_qty)

        # Deduct trading fee on partial close
        partial_notional = close_qty * price
        fee = partial_notional * config.TRADE_FEE_PERCENT
        pnl -= fee

        # Reduce position size
        pos.qty -= close_qty
        pos.realized_pnl += pnl

        # Zombie Position Fix: If qty is zero or dust, close it fully
        if pos.qty < 1e-8:
            pos.qty = 0
            pos.is_open = False
            if pos_id in self.positions:
                del self.positions[pos_id]
            log.info(f"SIM: Position {pos_id} finalized (qty ~0) after partial close.")

        # Update balance (release proportional margin + pnl)
        partial_notional_entry = close_qty * pos.entry_price
        partial_margin = partial_notional_entry / pos.leverage
        self.balance.used -= partial_margin
        self.balance.available += partial_margin + pnl
        self.balance.total += pnl

        log.info(f"SIM: Partial close {fraction*100:.0f}% of pos {pos_id} ({pos.symbol}) @ {price} | PnL: {pnl:.2f}")
        return pnl

    def _calc_liquidation_price(self, entry: float, side: str, leverage: float) -> float:
        # Simplified institutional formula
        if side == "long":
            return entry * (1 - (1 / leverage) + self.maintenance_margin)
        else:
            return entry * (1 + (1 / leverage) - self.maintenance_margin)

    def get_balance(self) -> SimulatedBalance:
        return self.balance

    def deduct_funding_costs(self, symbol: str, rate: float):
        """Deducts funding costs from all open positions for a symbol."""
        for pos in self.positions.values():
            if pos.is_open and pos.symbol == symbol:
                notional = pos.qty * self.last_prices.get(symbol, pos.entry_price)
                cost = notional * rate
                # For longs, positive rate = pay. For shorts, positive rate = receive.
                # Negate cost for shorts: positive rate means shorts receive (cost becomes negative/credit), 
                # negative rate means shorts pay (cost becomes positive/charge).
                if pos.side == "short": cost = -cost
                
                self.balance.available -= cost
                self.balance.total -= cost
                log.info(f"SIM: Funding cost for {symbol} ({pos.side}): {cost:.4f} (Rate: {rate:.4f})")
