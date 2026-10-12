import os
os.environ["MODE"] = "replay"
import asyncio
import pandas as pd
import logging
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, UTC, timezone
import config
from dataclasses import dataclass, field
from typing import Optional
# NOTE: This script runs the V10.0 backtest/replay engine.
# It is NOT part of the V11.0 live stack. Use main.py to run live engines.
# The archived bot logic lives in archive/bot_v10_backtest.py.
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "archive"))
from bot_v10_backtest import run_cycle, calculate_btc_macro_trend
from state_manager import EngineState, TradeEntry
from db_manager import DatabaseManager
from exchange_manager import ExchangeManager
from indicators import compute_indicators

# Setup basic logging for backtest
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("backtest")

@dataclass
class MockPosition:
    id: int
    symbol: str
    side: str
    qty: float
    entry_price: float
    sl: float
    tp: float
    regime: str
    phase: str
    quality: float
    notes: str
    partial_taken: bool = False
    sl_offset: float = 0.0
    sl_mode: str = "normal"
    tp_mode: str = "swing"
    trail_mode: str = "normal"
    unr: float = 0.0
    peak_price: float = 0.0
    mfe: float = 0.0
    realized_pnl: float = 0.0
    entry_fee: float = 0.0

_GLOBAL_POS_ID_COUNTER = 0

class MockExchangeManager:
    """Simulates CCXT behavior for multiple symbols - Optimized for Speed."""
    def __init__(self, symbols, initial_balance=None):
        self.symbols = symbols
        self.dfs = {}
        self.dfs_4h = {}
        self.mappings_4h = {}
        
        # Load and pre-process each symbol
        all_indices = []
        for symbol in symbols:
            # Normalize symbol name for file path: BTC/USDT -> btc_usdt
            safe_name = symbol.replace("/", "_").lower()
            csv_path = f"historical_{safe_name}.csv"
            
            # Fallback for the existing historical_1h.csv if only BTC is used
            if not os.path.exists(csv_path) and symbol == "BTC/USDT":
                csv_path = "historical_1h.csv"
                
            if not os.path.exists(csv_path):
                log.warning(f"Data file not found for {symbol}: {csv_path}. Skipping.")
                continue
                
            log.info(f"Loading {symbol} from {csv_path}...")
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df['timestamp_ms'] = df.index.astype('int64') // 10**6
            
            log.info(f"Pre-calculating indicators for {symbol}...")
            df = compute_indicators(df)
            self.dfs[symbol] = df
            all_indices.append(df.index)
            
            # Pre-calculate 4h
            df_4h = df.resample('4h', label='right', closed='right').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
                'ema20': 'last', 'ema50': 'last', 'ema200': 'last'
            }).dropna()
            df_4h['timestamp_ms'] = df_4h.index.astype('int64') // 10**6
            self.dfs_4h[symbol] = df_4h
            
            # Pre-calculate 4h mapping
            h4_ptr = 0
            h4_times = df_4h['timestamp_ms'].values
            mapping = []
            for t in df['timestamp_ms'].values:
                while h4_ptr < len(h4_times) - 1 and h4_times[h4_ptr+1] <= t:
                    h4_ptr += 1
                mapping.append(h4_ptr)
            self.mappings_4h[symbol] = mapping

        # Create a unified timeline (union of all timestamps)
        if not all_indices:
            raise FileNotFoundError("No historical data files found for any symbol.")
            
        self.timeline = pd.Index(sorted(set().union(*all_indices)))
        self.cursor_time = self.timeline[801] # Initial warmup
        
        self.balance = type('Balance', (), {'total': initial_balance or config.INITIAL_BALANCE})()
        self.positions = []
        self.sim = type('Sim', (), {'positions': self.positions})() 
        
    def fetch_df(self, symbol, timeframe, limit=200):
        df_base = self.dfs_4h[symbol] if timeframe == "4h" else self.dfs[symbol]
        
        # Find the integer location of the current cursor_time in this symbol's DF
        # We use 'ffill' logic: get the latest candle available AT OR BEFORE cursor_time
        try:
            loc = df_base.index.get_indexer([self.cursor_time], method='pad')[0]
            if loc == -1: return pd.DataFrame() # No data yet for this symbol
            
            if timeframe == "4h":
                # For 4H, we use the pre-calculated mapping from the 1H loc
                loc_1h = self.dfs[symbol].index.get_indexer([self.cursor_time], method='pad')[0]
                ptr = self.mappings_4h[symbol][loc_1h]
                return self.dfs_4h[symbol].iloc[max(0, ptr - limit + 1) : ptr + 1]
                
            return df_base.iloc[max(0, loc - limit + 1) : loc + 1]
        except:
            return pd.DataFrame()

    def fetch_ohlcv(self, symbol, timeframe, limit=200):
        df = self.fetch_df(symbol, timeframe, limit)
        return df.values.tolist()

    def fetch_bbo(self, symbol):
        df = self.dfs[symbol]
        loc = df.index.get_indexer([self.cursor_time], method='pad')[0]
        row = df.iloc[loc]
        return {"bid": row['close'], "ask": row['close']}

    def update_sim_price(self, symbol, price):
        self.current_price = float(price)
        # Update simulation price and trigger active position checks for the symbol
        for p in self.positions:
            if p.symbol == symbol:
                if p.side == "long":
                    if price > p.peak_price:
                        p.peak_price = price
                else:
                    if price < p.peak_price:
                        p.peak_price = price

    def get_positions_by_symbol(self, symbol):
        return [p for p in self.positions if p.symbol == symbol]

    def open_sim_pos(self, symbol, side, qty, price, leverage, sl, tp, regime, phase, quality, notes):
        global _GLOBAL_POS_ID_COUNTER
        _GLOBAL_POS_ID_COUNTER += 1
        pos_id = _GLOBAL_POS_ID_COUNTER
        
        # Apply Slippage: Long entries are higher, Short entries are lower
        slippage_price = price * (1 + 0.0005 if side == 'long' else 1 - 0.0005)
        
        # Calculate Entry Fee (0.1% of Notional)
        entry_fee = qty * slippage_price * 0.001
        self.balance.total -= entry_fee
        
        pos = MockPosition(
            id=pos_id, symbol=symbol, side=side, qty=qty, 
            entry_price=slippage_price, sl=sl, tp=tp, regime=regime, 
            phase=phase, quality=quality, notes=notes,
            peak_price=slippage_price,
            entry_fee=entry_fee
        )
        self.positions.append(pos)
        return pos

    def close_sim_pos(self, pos_id, price):
        for i, p in enumerate(self.positions):
            if p.id == pos_id:
                # Apply Slippage: Long exits are lower, Short exits are higher
                slippage_price = price * (1 - 0.0005 if p.side == 'long' else 1 + 0.0005)
                
                # Gross PnL based on slippage prices
                gross_pnl = (slippage_price - p.entry_price if p.side == 'long' else p.entry_price - slippage_price) * p.qty
                
                # Exit Fee (0.1% of Notional)
                exit_fee = p.qty * slippage_price * 0.001
                
                # Final Net PnL (includes entry fee already deducted from balance)
                net_pnl = gross_pnl - exit_fee
                
                # Update running balance
                self.balance.total += net_pnl
                
                # Hard Liquidation Check
                if self.balance.total <= 0:
                    self.balance.total = 0
                    log.error("CRITICAL: Portfolio Liquidated. Halting Simulation.")
                
                closed = self.positions.pop(i)
                # realized_pnl reported to logs must include the entry fee for reconciliation
                closed.realized_pnl = net_pnl - p.entry_fee
                return closed
        return None

    def close_partial_sim_pos(self, pos_id, fraction, price):
        for p in self.positions:
            if p.id == pos_id:
                close_qty = p.qty * fraction
                
                # Apply Slippage on partial exit
                slippage_price = price * (1 - 0.0005 if p.side == 'long' else 1 + 0.0005)
                
                # Gross PnL on fraction
                gross_pnl = (slippage_price - p.entry_price if p.side == 'long' else p.entry_price - slippage_price) * close_qty
                
                # Exit Fee on fraction
                exit_fee = close_qty * slippage_price * 0.001
                
                # Partial Entry Fee correction
                partial_entry_fee = p.entry_fee * fraction
                p.entry_fee -= partial_entry_fee
                
                # Net PnL for this partial exit
                net_pnl = gross_pnl - exit_fee
                
                self.balance.total += net_pnl
                
                if self.balance.total <= 0:
                    self.balance.total = 0
                    log.error("CRITICAL: Portfolio Liquidated on Partial TP. Halting.")
                
                p.qty -= close_qty
                # Return net pnl minus the entry fee for reconciliation
                return net_pnl - partial_entry_fee
        return None

    def update_mock_data(self, current_time):
        """Update peak prices and MFE for all open positions based on current time."""
        self.cursor_time = current_time
        for p in self.positions:
            df = self.dfs[p.symbol]
            loc = df.index.get_indexer([current_time], method='pad')[0]
            price = df.iloc[loc]['close']
            if p.side == "long":
                if price > p.peak_price:
                    p.peak_price = price
                    p.mfe = max(p.mfe, (price - p.entry_price) / p.entry_price * 100)
            else:
                if price < p.peak_price:
                    p.peak_price = price
                    p.mfe = max(p.mfe, (p.entry_price - price) / p.entry_price * 100)

async def main():
    print("============================================================")
    print("   LAUNCHING SQUADRON BACKTEST SIMULATOR [MULTI-ASSET]")
    print("============================================================")
    
    # Purge old logs to avoid ParserErrors
    for f in ["trade_history.csv", "trades.csv", "trading_data.db"]:
        if os.path.exists(f): os.remove(f)
        
    db = DatabaseManager()
    now_iso = datetime.now(UTC).isoformat()
    with db._get_conn() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM cycles")
        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES ('balance_total', ?, ?)", (str(config.INITIAL_BALANCE), now_iso))
        conn.execute("REPLACE INTO system_state (key, value, updated_at) VALUES ('session_start', ?, ?)", (now_iso, now_iso))
        conn.commit()
    log.info(f"Database Reset: Fresh session initialized with ${config.INITIAL_BALANCE:,.2f}")

    # Initialize multi-symbol components
    mock_em = MockExchangeManager(symbols=config.SYMBOLS, initial_balance=config.INITIAL_BALANCE)
    state = EngineState(sim_instance=mock_em)
    db.save_state("kill_switch", "READY")
    
    total_steps = len(mock_em.timeline)
    log.info(f"Starting Multi-Asset Simulation: {total_steps} cycles on synchronized timeline...")

    try:
        # Filter timeline to user's requested period: 2019-01-01 to 2024-12-31
        study_start = pd.Timestamp("2019-01-01", tz="UTC")
        study_end = pd.Timestamp("2024-12-31", tz="UTC")
        study_timeline = [t for t in mock_em.timeline if study_start <= t <= study_end]
        
        log.info(f"Filtered Timeline: {len(study_timeline)} study cycles (2019-2024).")

        for i, candle_time in enumerate(study_timeline):
            try:
                mock_em.update_mock_data(candle_time)
                
                # 1. BTC Macro remains the anchor
                await calculate_btc_macro_trend(mock_em, state, candle_time=candle_time)
                
                # 2. Fire Squadron Cycles in parallel
                tasks = [run_cycle(symbol, mock_em, state, candle_time=candle_time) for symbol in config.SYMBOLS]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for res in results:
                    if isinstance(res, Exception):
                        log.error(f"Symbol Cycle Error: {res}")
                    elif res:
                        state.cycle_count += 1
                
                state.update_metrics()
                
            except Exception as e:
                log.error(f"Timeline Cycle Error at {candle_time}: {e}")
                break
    finally:
        state.close()
        try:
            from ai_engine import get_llm
            await get_llm().close()
        except Exception as e: log.warning(f"Failed to gracefully close LLM client: {e}")
        
        # --- SQUADRON PERFORMANCE REPORTING ---
        print("\n" + "="*60)
        print("   SQUADRON (V9.0) MASTER STUDY PERFORMANCE REPORT")
        print("   Period: 2019-01-01 to 2024-12-31")
        print("="*60)
        
        # 1. Symbol Performance Table
        trades_df = pd.read_csv("trade_history.csv")
        if not trades_df.empty:
            print(f"\n{'SYMBOL':<10} | {'WIN%':<8} | {'PF':<6} | {'TOTAL PNL':<12}")
            print("-" * 45)
            for sym in config.SYMBOLS:
                sym_trades = trades_df[trades_df['symbol'] == sym]
                if sym_trades.empty: continue
                
                wins = len(sym_trades[sym_trades['pnl'] > 0])
                total = len(sym_trades)
                wr = (wins / total * 100) if total > 0 else 0
                
                gross_profits = sym_trades[sym_trades['pnl'] > 0]['pnl'].sum()
                gross_losses = abs(sym_trades[sym_trades['pnl'] < 0]['pnl'].sum())
                pf = (gross_profits / gross_losses) if gross_losses > 0 else 99.9
                
                total_pnl = sym_trades['pnl'].sum()
                print(f"{sym:<10} | {wr:>6.1f}% | {pf:>5.2f} | ${total_pnl:>9.2f}")
        
        # 2. Portfolio Summary
        net_pnl = trades_df['pnl'].sum() if not trades_df.empty else 0
        expected_bal = config.INITIAL_BALANCE + net_pnl
        actual_bal = state.get_total_balance()
        
        print("\n" + "-"*60)
        print(f"FINAL BALANCE:  ${actual_bal:,.2f}")
        print(f"NET PNL SUM:    ${net_pnl:,.2f}")
        print(f"PNL RECONCILE:  {'MATCH' if abs(expected_bal - actual_bal) < 1.0 else 'DISCREPANCY'}")
        print(f"MAX DRAWDOWN:   {state.max_drawdown:.2f}%")
        print(f"TOTAL TRADES:   {len(trades_df) if not trades_df.empty else 0}")
        print("-"*60)
        
        # 3. Highlight 'Big Winner'
        if not trades_df.empty:
            winners = trades_df[trades_df['pnl'] > 0]
            if not winners.empty:
                big_winner = winners.loc[winners['pnl'].idxmax()]
                print(f"\n[WINNER] THE BIG WINNER:")
                print(f"  Symbol:   {big_winner['symbol']}")
                print(f"  PnL:      ${big_winner['pnl']:.2f}")
                print(f"  Quality:  {big_winner['quality']}")
                print(f"  Notes:    {big_winner['notes']}")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
