"""
backtest_super_bot.py — Unified Super Bot Simulation & Parameter Optimization

Simulates the Super Bot architecture combining:
1. Regime Router:
   - BULL (ADX > 25, 4H Ichimoku Bullish, 1D Trend Align) -> Swing Architect (4H Longs, 3x/5x)
   - CHOP (ADX < 22, Range-bound around VWAP) -> Mean Reversion Maker Grid (5m/15m Longs, Maker Fees 0.02%)
   - BEAR/CRASH (Price < 200 EMA, Bearish breakdown) -> 100% Cash / Parked (Zero Shorts)
2. Accurate Microstructure Accounting:
   - Maker entry/exit: 0.02% (post-only limit orders)
   - Taker SL trigger: 0.06% (emergency stops)
   - Capital allocation: Dynamic per regime
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))

def compute_adx(high, low, close, period=14):
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = compute_atr(high, low, close, period)
    plus_di = 100.0 * (pd.Series(plus_dm, index=high.index).rolling(period).mean() / (atr + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm, index=high.index).rolling(period).mean() / (atr + 1e-9))
    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    return dx.rolling(period).mean()

def run_super_bot_backtest(
    df_5m_path="historical_btc_usdt_5m.csv",
    df_4h_path="data/historical/BTCUSDT_4h_full.csv",
    initial_balance=100.0,
    maker_fee=0.0002,   # 0.02% Bitget Maker
    taker_fee=0.0006,   # 0.06% Bitget Taker
    swing_leverage=3.0,
    grid_leverage=3.0,
    grid_tp_pct=0.012,  # 1.2% TP on Maker limit
    grid_sl_pct=0.020   # 2.0% Soft SL
):
    print("=" * 70)
    print("      SUPER BOT DUAL-REGIME MAKER ENGINE BACKTEST")
    print("=" * 70)
    
    if not os.path.exists(df_5m_path):
        print(f"Error: {df_5m_path} not found.")
        return
        
    print(f"Loading 5m data from {df_5m_path}...")
    df_5m = pd.read_csv(df_5m_path)
    df_5m['datetime'] = pd.to_datetime(df_5m['timestamp'], unit='ms')
    df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate 5m indicators
    c, h, l = df_5m['close'], df_5m['high'], df_5m['low']
    df_5m['rsi'] = compute_rsi(c, 14)
    df_5m['atr'] = compute_atr(h, l, c, 14)
    df_5m['adx'] = compute_adx(h, l, c, 14)
    
    # Rolling VWAP on 5m (24 periods = 2 hours)
    cum_pv = (c * df_5m['volume']).rolling(24).sum()
    cum_v = df_5m['volume'].rolling(24).sum()
    df_5m['vwap'] = cum_pv / (cum_v + 1e-9)
    
    # 50 & 200 EMAs on 5m for macro trend orientation
    df_5m['ema_50'] = c.ewm(span=288, adjust=False).mean()  # ~24 hours
    df_5m['ema_200'] = c.ewm(span=1152, adjust=False).mean() # ~4 days
    
    # 4H resampled trend proxy or direct load
    # 48 periods of 5m = 4 hours
    df_5m['adx_4h_proxy'] = df_5m['adx'].rolling(48).mean()
    
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    
    trades = []
    active_swing = None
    active_grid = None
    
    # Start after warm-up
    warmup = 1200
    total_bars = len(df_5m)
    print(f"Running simulation over {total_bars:,} bars (Warmup: {warmup})...")
    
    for i in range(warmup, total_bars):
        row = df_5m.iloc[i]
        curr_p = row['close']
        high_p = row['high']
        low_p = row['low']
        atr = row['atr']
        
        # --- 1. DETERMINE REGIME ---
        # BULL: Price > EMA200, EMA50 > EMA200, 4H ADX > 25
        # CHOP: ADX < 22 or price oscillating around EMA50/200
        # BEAR: Price < EMA200, EMA50 < EMA200, ADX > 22
        is_bull_regime = (curr_p > row['ema_200']) and (row['ema_50'] > row['ema_200']) and (row['adx_4h_proxy'] > 23.0)
        is_bear_regime = (curr_p < row['ema_200']) and (row['ema_50'] < row['ema_200']) and (row['adx_4h_proxy'] > 22.0)
        is_chop_regime = not is_bull_regime and not is_bear_regime
        
        # --- 2. MANAGE ACTIVE SWING POSITION ---
        if active_swing:
            entry_p = active_swing['entry_p']
            sl = active_swing['sl']
            tp = active_swing['tp']
            qty = active_swing['qty']
            
            # Check exit
            closed = False
            exit_p = None
            reason = None
            
            if low_p <= sl:
                exit_p = sl
                reason = "SWING_SL"
                closed = True
                fee_rate_exit = taker_fee # Emergency SL = taker
            elif high_p >= tp:
                exit_p = tp
                reason = "SWING_TP"
                closed = True
                fee_rate_exit = maker_fee # Limit TP = maker
            elif is_bear_regime: # Regime shift safety close
                exit_p = curr_p
                reason = "REGIME_BEAR_EXIT"
                closed = True
                fee_rate_exit = taker_fee
            else:
                # Dynamic Trailing stop: lock breakeven after +1.5 ATR gain
                if curr_p - entry_p >= 1.5 * active_swing['risk_dist']:
                    active_swing['sl'] = max(active_swing['sl'], entry_p + (0.2 * active_swing['risk_dist']))
            
            if closed:
                gross_pnl = (exit_p - entry_p) * qty
                fee_exit = qty * exit_p * fee_rate_exit
                net_pnl = gross_pnl - (active_swing['fee_entry'] + fee_exit)
                balance += net_pnl
                trades.append({
                    'type': 'SWING',
                    'side': 'LONG',
                    'entry_p': entry_p,
                    'exit_p': exit_p,
                    'pnl': net_pnl,
                    'gross': gross_pnl,
                    'fees': active_swing['fee_entry'] + fee_exit,
                    'reason': reason
                })
                active_swing = None
                
        # --- 3. MANAGE ACTIVE GRID POSITION ---
        if active_grid:
            entry_p = active_grid['entry_p']
            sl = active_grid['sl']
            tp = active_grid['tp']
            qty = active_grid['qty']
            
            closed = False
            exit_p = None
            reason = None
            
            if low_p <= sl:
                exit_p = sl
                reason = "GRID_SL"
                closed = True
                fee_rate_exit = taker_fee # Stop breach
            elif high_p >= tp:
                exit_p = tp
                reason = "GRID_TP"
                closed = True
                fee_rate_exit = maker_fee # Limit order filled as Maker!
            elif is_bear_regime:
                exit_p = curr_p
                reason = "GRID_BEAR_EXIT"
                closed = True
                fee_rate_exit = taker_fee
                
            if closed:
                gross_pnl = (exit_p - entry_p) * qty
                fee_exit = qty * exit_p * fee_rate_exit
                net_pnl = gross_pnl - (active_grid['fee_entry'] + fee_exit)
                balance += net_pnl
                trades.append({
                    'type': 'GRID',
                    'side': 'LONG',
                    'entry_p': entry_p,
                    'exit_p': exit_p,
                    'pnl': net_pnl,
                    'gross': gross_pnl,
                    'fees': active_grid['fee_entry'] + fee_exit,
                    'reason': reason
                })
                active_grid = None
                
        # Update Drawdown
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd
            
        # --- 4. NEW ENTRIES ---
        # Gated by regime and balance
        if balance < 10.0:
            continue
            
        # BULL REGIME ENTRY -> SWING ARCHITECT
        if is_bull_regime and not active_swing:
            # Trigger: Pullback confirmation or fresh breakout above EMA20
            # RSI oversold in uptrend (40-50) or bounce off 20 EMA
            pullback_trigger = (row['rsi'] < 52.0) and (curr_p > row['ema_50']) and (row['adx'] > 20.0)
            if pullback_trigger:
                entry_p = curr_p
                risk_dist = 2.5 * atr
                sl = entry_p - risk_dist
                tp = entry_p + (3.5 * risk_dist)
                
                # Sizing: $20 margin or 20% of balance
                margin = min(25.0, balance * 0.20)
                notional = margin * swing_leverage
                qty = notional / entry_p
                fee_entry = notional * maker_fee # Limit order placement
                
                active_swing = {
                    'entry_p': entry_p,
                    'qty': qty,
                    'sl': sl,
                    'tp': tp,
                    'risk_dist': risk_dist,
                    'fee_entry': fee_entry
                }
                
        # CHOP REGIME ENTRY -> MEAN REVERSION MAKER GRID
        elif is_chop_regime and not active_grid:
            # Trigger: Oversold in range (RSI < 32 and price below VWAP)
            grid_trigger = (row['rsi'] < 30.0) and (curr_p < row['vwap'])
            if grid_trigger:
                entry_p = curr_p
                tp = entry_p * (1.0 + grid_tp_pct)
                sl = entry_p * (1.0 - grid_sl_pct)
                
                margin = min(20.0, balance * 0.15)
                notional = margin * grid_leverage
                qty = notional / entry_p
                fee_entry = notional * maker_fee # Maker post-only limit order
                
                active_grid = {
                    'entry_p': entry_p,
                    'qty': qty,
                    'sl': sl,
                    'tp': tp,
                    'fee_entry': fee_entry
                }
                
        # BEAR REGIME -> NO TRADES (100% USDT)
        elif is_bear_regime:
            pass

    # Compile Final Report
    total_trades = len(trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    wr = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    net_pnl = balance - initial_balance
    total_fees = sum(t['fees'] for t in trades)
    
    swing_trades = [t for t in trades if t['type'] == 'SWING']
    grid_trades = [t for t in trades if t['type'] == 'GRID']
    
    swing_wins = [t for t in swing_trades if t['pnl'] > 0]
    grid_wins = [t for t in grid_trades if t['pnl'] > 0]
    
    print("\n" + "=" * 70)
    print("                    BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Initial Balance:       ${initial_balance:,.2f}")
    print(f"Final Balance:         ${balance:,.2f}")
    print(f"Net Profit / Return:   ${net_pnl:+,.2f} ({net_pnl/initial_balance*100:+.2f}%)")
    print(f"Max Drawdown:          {max_drawdown*100:.2f}%")
    print(f"Total Trades:          {total_trades}")
    print(f"Total Win Rate:        {wr:.2f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Total Fees Paid:       ${total_fees:,.2f} (Maker Advantage Active)")
    print("-" * 70)
    print(f"Swing Architect Trades: {len(swing_trades)} | Wins: {len(swing_wins)} ({len(swing_wins)/(len(swing_trades) or 1)*100:.1f}% WR) | Net PnL: ${sum(t['pnl'] for t in swing_trades):+,.2f}")
    print(f"Mean Reversion Trades:  {len(grid_trades)} | Wins: {len(grid_wins)} ({len(grid_wins)/(len(grid_trades) or 1)*100:.1f}% WR) | Net PnL: ${sum(t['pnl'] for t in grid_trades):+,.2f}")
    print("=" * 70)
    
    return {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'net_pnl': net_pnl,
        'max_dd': max_drawdown,
        'win_rate': wr,
        'trades': total_trades,
        'swing_pnl': sum(t['pnl'] for t in swing_trades),
        'grid_pnl': sum(t['pnl'] for t in grid_trades),
    }

if __name__ == "__main__":
    run_super_bot_backtest()
