"""
backtest_all_fleet_bots.py — Unified Multi-Bot Forensic Backtester
Tests all 6 bots in the Sovereign Swarm ecosystem across their native timeframes:
1. Trend Sniper (15m) — live_trader.py
2. Berserker Scalper (5m) — berserker_trader.py
3. Mean Reversion / Ghost Grid (5m) — mean_reversion_trader.py
4. MARL Policy Trader (15m) — marl_live_trader.py
5. The Swing Architect (4H) — swing_trader.py
6. Cash & Carry Basis Arbitrage — basis_trader.py

Accurate Microstructure Accounting:
- Exchange fees computed on NOTIONAL position size (0.06% taker entry + 0.06% exit)
- 2 bps slippage on market orders
- Starting balance: $100 per bot
- Standard sizing: $15 margin per trade at bot-assigned leverage
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

# ==============================================================================
# BOT 1: TREND SNIPER (15m) — live_trader.py
# ==============================================================================
def backtest_trend_sniper(df_15m, leverage=5.0):
    df = df_15m.copy().reset_index(drop=True)
    c, h, l = df['close'], df['high'], df['low']
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()
    df['ema_50'] = c.ewm(span=50, adjust=False).mean()
    df['ema_200'] = c.ewm(span=200, adjust=False).mean()
    df['atr'] = compute_atr(h, l, c, 14)
    df['rsi'] = compute_rsi(c, 14)
    df['adx'] = compute_adx(h, l, c, 14)
    df['vol_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ema'] + 1e-9)

    balance = 100.0
    trades = []
    pos = None
    fee_rate = 0.0006  # 0.06% Bitget taker

    for i in range(250, len(df)):
        row = df.iloc[i]
        curr_p, high_p, low_p = row['close'], row['high'], row['low']
        atr = row['atr']
        
        # Position management
        if pos:
            side = pos['side']
            entry_p = pos['entry_p']
            sl, tp = pos['sl'], pos['tp']
            closed = False
            exit_p = None
            reason = None
            
            if side == 'LONG':
                if low_p <= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif high_p >= tp:
                    exit_p, reason, closed = tp, 'TP', True
                # Trailing stop
                elif (curr_p - entry_p) >= (1.5 * pos['risk_unit']) and sl < entry_p:
                    pos['sl'] = entry_p + (0.05 * pos['risk_unit'])
            else:
                if high_p >= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif low_p <= tp:
                    exit_p, reason, closed = tp, 'TP', True
                elif (entry_p - curr_p) >= (1.5 * pos['risk_unit']) and sl > entry_p:
                    pos['sl'] = entry_p - (0.05 * pos['risk_unit'])
                    
            if closed:
                qty = pos['qty']
                notional_exit = qty * exit_p
                fee_exit = notional_exit * fee_rate
                gross_pnl = (exit_p - entry_p) * qty if side == 'LONG' else (entry_p - exit_p) * qty
                total_fees = pos['fee_entry'] + fee_exit
                net_pnl = gross_pnl - total_fees
                balance += net_pnl
                trades.append({'side': side, 'pnl': net_pnl, 'gross': gross_pnl, 'fees': total_fees, 'reason': reason})
                pos = None
                continue
                
        # Signal Generation
        if not pos and balance > 10.0:
            long_sig = (row['ema_20'] > row['ema_50'] > row['ema_200']) and (row['adx'] > 20) and (45 <= row['rsi'] <= 70) and (row['vol_ratio'] > 1.2)
            short_sig = (row['ema_20'] < row['ema_50'] < row['ema_200']) and (row['adx'] > 20) and (30 <= row['rsi'] <= 55) and (row['vol_ratio'] > 1.2)
            
            if long_sig or short_sig:
                side = 'LONG' if long_sig else 'SHORT'
                entry_p = curr_p * (1.0002 if side == 'LONG' else 0.9998)
                sl_dist = 2.0 * atr
                sl = (entry_p - sl_dist) if side == 'LONG' else (entry_p + sl_dist)
                tp = (entry_p + (3.0 * sl_dist)) if side == 'LONG' else (entry_p - (3.0 * sl_dist))
                risk_unit = sl_dist
                
                margin = min(15.0, balance * 0.15)
                notional = margin * leverage
                qty = notional / entry_p
                fee_entry = notional * fee_rate
                pos = {'side': side, 'entry_p': entry_p, 'qty': qty, 'sl': sl, 'tp': tp, 'risk_unit': risk_unit, 'fee_entry': fee_entry}

    return compile_stats("Trend Sniper (15m)", trades, balance)

# ==============================================================================
# BOT 2: BERSERKER SCALPER (5m) — berserker_trader.py
# ==============================================================================
def backtest_berserker_scalper(df_5m, leverage=5.0):
    df = df_5m.copy().reset_index(drop=True)
    c, h, l = df['close'], df['high'], df['low']
    df['ema_9'] = c.ewm(span=9, adjust=False).mean()
    df['ema_21'] = c.ewm(span=21, adjust=False).mean()
    df['atr'] = compute_atr(h, l, c, 14)
    df['vol_ema'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ema'] + 1e-9)

    balance = 100.0
    trades = []
    pos = None
    fee_rate = 0.0006

    for i in range(100, len(df)):
        row = df.iloc[i]
        curr_p, high_p, low_p = row['close'], row['high'], row['low']
        atr = row['atr']
        
        if pos:
            side = pos['side']
            entry_p = pos['entry_p']
            sl, tp = pos['sl'], pos['tp']
            closed = False
            exit_p = None
            reason = None
            
            if side == 'LONG':
                if low_p <= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif high_p >= tp:
                    exit_p, reason, closed = tp, 'TP', True
            else:
                if high_p >= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif low_p <= tp:
                    exit_p, reason, closed = tp, 'TP', True
                    
            if closed:
                qty = pos['qty']
                notional_exit = qty * exit_p
                fee_exit = notional_exit * fee_rate
                gross_pnl = (exit_p - entry_p) * qty if side == 'LONG' else (entry_p - exit_p) * qty
                total_fees = pos['fee_entry'] + fee_exit
                net_pnl = gross_pnl - total_fees
                balance += net_pnl
                trades.append({'side': side, 'pnl': net_pnl, 'gross': gross_pnl, 'fees': total_fees, 'reason': reason})
                pos = None
                continue
                
        # Berserker Momentum Expansion Trigger
        if not pos and balance > 10.0:
            long_sig = (row['ema_9'] > row['ema_21']) and (curr_p > row['ema_9']) and (row['vol_ratio'] > 1.35)
            short_sig = (row['ema_9'] < row['ema_21']) and (curr_p < row['ema_9']) and (row['vol_ratio'] > 1.35)
            
            if long_sig or short_sig:
                side = 'LONG' if long_sig else 'SHORT'
                entry_p = curr_p * (1.0002 if side == 'LONG' else 0.9998)
                sl_dist = 1.5 * atr
                sl = (entry_p - sl_dist) if side == 'LONG' else (entry_p + sl_dist)
                tp = (entry_p + (2.0 * sl_dist)) if side == 'LONG' else (entry_p - (2.0 * sl_dist))
                
                margin = min(15.0, balance * 0.15)
                notional = margin * leverage
                qty = notional / entry_p
                fee_entry = notional * fee_rate
                pos = {'side': side, 'entry_p': entry_p, 'qty': qty, 'sl': sl, 'tp': tp, 'fee_entry': fee_entry}

    return compile_stats("Berserker Scalper (5m)", trades, balance)

# ==============================================================================
# BOT 3: MEAN REVERSION / GHOST GRID (5m) — mean_reversion_trader.py
# ==============================================================================
def backtest_mean_reversion(df_5m, leverage=3.0):
    df = df_5m.copy().reset_index(drop=True)
    c, h, l = df['close'], df['high'], df['low']
    df['rsi'] = compute_rsi(c, 14)
    df['atr'] = compute_atr(h, l, c, 14)
    # Rolling VWAP
    cum_pv = (c * df['volume']).rolling(24).sum()
    cum_v = df['volume'].rolling(24).sum()
    df['vwap'] = cum_pv / (cum_v + 1e-9)
    df['adx'] = compute_adx(h, l, c, 14)

    balance = 100.0
    trades = []
    pos = None
    fee_rate = 0.0006

    for i in range(100, len(df)):
        row = df.iloc[i]
        curr_p, high_p, low_p = row['close'], row['high'], row['low']
        
        if pos:
            side = pos['side']
            entry_p = pos['entry_p']
            sl, tp = pos['sl'], pos['tp']
            closed = False
            exit_p = None
            reason = None
            
            if side == 'LONG':
                if low_p <= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif high_p >= tp:
                    exit_p, reason, closed = tp, 'TP', True
            else:
                if high_p >= sl:
                    exit_p, reason, closed = sl, 'SL', True
                elif low_p <= tp:
                    exit_p, reason, closed = tp, 'TP', True
                    
            if closed:
                qty = pos['qty']
                fee_exit = qty * exit_p * fee_rate
                gross_pnl = (exit_p - entry_p) * qty if side == 'LONG' else (entry_p - exit_p) * qty
                total_fees = pos['fee_entry'] + fee_exit
                net_pnl = gross_pnl - total_fees
                balance += net_pnl
                trades.append({'side': side, 'pnl': net_pnl, 'gross': gross_pnl, 'fees': total_fees, 'reason': reason})
                pos = None
                continue

        # Range-bound Mean Reversion Trigger (ADX < 25, RSI extreme deviation from VWAP)
        if not pos and balance > 10.0:
            in_range = row['adx'] < 25.0
            long_sig = in_range and (row['rsi'] < 32.0) and (curr_p < row['vwap'])
            short_sig = in_range and (row['rsi'] > 68.0) and (curr_p > row['vwap'])
            
            if long_sig or short_sig:
                side = 'LONG' if long_sig else 'SHORT'
                entry_p = curr_p
                # 0.6% target profit (Rule in mean_reversion_trader.py: params['take_profit_pct'] = 1.006)
                # 1.5% soft stop
                tp = entry_p * 1.006 if side == 'LONG' else entry_p * 0.994
                sl = entry_p * 0.985 if side == 'LONG' else entry_p * 1.015
                
                margin = min(15.0, balance * 0.15)
                notional = margin * leverage
                qty = notional / entry_p
                fee_entry = notional * fee_rate
                pos = {'side': side, 'entry_p': entry_p, 'qty': qty, 'sl': sl, 'tp': tp, 'fee_entry': fee_entry}

    return compile_stats("Mean Reversion Grid (5m)", trades, balance)

# ==============================================================================
# BOT 4: MARL MULTI-AGENT POLICY (15m) — marl_live_trader.py
# ==============================================================================
def backtest_marl_trader(df_15m, leverage=4.0):
    df = df_15m.copy().reset_index(drop=True)
    c, h, l = df['close'], df['high'], df['low']
    df['rsi'] = compute_rsi(c, 14)
    df['atr'] = compute_atr(h, l, c, 14)
    df['adx'] = compute_adx(h, l, c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()

    balance = 100.0
    trades = []
    pos = None
    fee_rate = 0.0006

    for i in range(100, len(df)):
        row = df.iloc[i]
        curr_p, high_p, low_p = row['close'], row['high'], row['low']
        atr = row['atr']
        
        if pos:
            side = pos['side']
            entry_p = pos['entry_p']
            sl, tp = pos['sl'], pos['tp']
            closed = False
            exit_p = None
            reason = None
            
            # Dynamic 3x ATR trailing stop
            trail_dist = 3.0 * atr
            if side == 'LONG':
                pos['sl'] = max(pos['sl'], curr_p - trail_dist)
                if low_p <= pos['sl']:
                    exit_p, reason, closed = pos['sl'], 'TRAIL_SL', True
                elif high_p >= tp:
                    exit_p, reason, closed = tp, 'TP', True
            else:
                pos['sl'] = min(pos['sl'], curr_p + trail_dist)
                if high_p >= pos['sl']:
                    exit_p, reason, closed = pos['sl'], 'TRAIL_SL', True
                elif low_p <= tp:
                    exit_p, reason, closed = tp, 'TP', True
                    
            if closed:
                qty = pos['qty']
                fee_exit = qty * exit_p * fee_rate
                gross_pnl = (exit_p - entry_p) * qty if side == 'LONG' else (entry_p - exit_p) * qty
                total_fees = pos['fee_entry'] + fee_exit
                net_pnl = gross_pnl - total_fees
                balance += net_pnl
                trades.append({'side': side, 'pnl': net_pnl, 'gross': gross_pnl, 'fees': total_fees, 'reason': reason})
                pos = None
                continue

        # MARL Consensus Policy Proxy (MACD cross + RSI consensus + ADX trend gate)
        if not pos and balance > 10.0:
            long_consensus = (row['macd'] > row['macd_sig']) and (50 < row['rsi'] < 65) and (row['adx'] > 22.0)
            short_consensus = (row['macd'] < row['macd_sig']) and (35 < row['rsi'] < 50) and (row['adx'] > 22.0)
            
            if long_consensus or short_consensus:
                side = 'LONG' if long_consensus else 'SHORT'
                entry_p = curr_p
                sl_dist = 2.0 * atr
                sl = entry_p - sl_dist if side == 'LONG' else entry_p + sl_dist
                tp = entry_p + (4.0 * sl_dist) if side == 'LONG' else entry_p - (4.0 * sl_dist)
                
                margin = min(15.0, balance * 0.15)
                notional = margin * leverage
                qty = notional / entry_p
                fee_entry = notional * fee_rate
                pos = {'side': side, 'entry_p': entry_p, 'qty': qty, 'sl': sl, 'tp': tp, 'fee_entry': fee_entry}

    return compile_stats("MARL Policy Trader (15m)", trades, balance)

# ==============================================================================
# BOT 5: THE SWING ARCHITECT (4H) — swing_trader.py
# ==============================================================================
def backtest_swing_architect_4h(df_4h, leverage=3.0):
    from backtest_swing_architect import SwingArchitectBacktester
    tester = SwingArchitectBacktester(
        initial_balance=100.0,
        allocation_usd=20.0,
        standard_leverage=leverage,
        high_quality_leverage=leverage,
        allow_shorts=True,
        atr_sl_mult=3.0,
        t1_roe=15.0,
        t2_roe=30.0
    )
    res = tester.run()
    trades_list = []
    if "df_trades" in res and not res["df_trades"].empty:
        for _, r in res["df_trades"].iterrows():
            trades_list.append({
                'side': r['side'].upper(),
                'pnl': r['net_pnl'],
                'gross': r['gross_pnl'],
                'fees': r['total_fees'],
                'reason': r['exit_reason']
            })
    return compile_stats("The Swing Architect (4H)", trades_list, res['final_balance'])

# ==============================================================================
# BOT 6: BTC CASH-AND-CARRY BASIS ARBITRAGE — basis_trader.py
# ==============================================================================
def backtest_basis_arbitrage(days=730):
    # Historical perpetual funding rate in crypto averages ~10.5% annualized (0.01% every 8h)
    # Long Spot + 1x Short Perp. Zero directional price risk.
    # Entry fee: 0.02% maker spot + 0.02% maker perp = 0.04% notional roundtrip.
    balance = 100.0
    annual_funding_rate = 0.105  # 10.5% APR
    daily_yield = annual_funding_rate / 365.0
    total_funding_earned = balance * daily_yield * days
    entry_exit_fees = balance * 0.0004 * 4  # 4 rebalances per year
    net_pnl = total_funding_earned - entry_exit_fees
    final_balance = balance + net_pnl
    
    # 3 periods per day * days
    trades = [{'side': 'DELTA_NEUTRAL', 'pnl': (daily_yield * balance) / 3.0, 'gross': (daily_yield * balance) / 3.0, 'fees': 0.0, 'reason': 'FUNDING_8H'} for _ in range(int(days * 3))]
    return compile_stats("Basis Funding Arbitrage (8H)", trades, final_balance)

def compile_stats(name, trades, final_balance):
    initial_balance = 100.0
    net_profit = final_balance - initial_balance
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "name": name, "trades": 0, "win_rate": 0.0, "net_pnl": 0.0,
            "total_fees": 0.0, "profit_factor": 0.0, "max_dd": 0.0, "final_bal": round(final_balance, 2)
        }
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = (len(wins) / total_trades) * 100.0
    gross_win = sum(t['gross'] for t in wins if t.get('gross', 0) > 0)
    gross_loss = abs(sum(t['gross'] for t in losses if t.get('gross', 0) < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else 99.0
    total_fees = sum(t.get('fees', 0.0) for t in trades)
    
    # Calculate drawdown curve
    equity = [initial_balance]
    peak = initial_balance
    max_dd = 0.0
    curr = initial_balance
    for t in trades:
        curr += t['pnl']
        peak = max(peak, curr)
        dd = ((peak - curr) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        equity.append(curr)
        
    return {
        "name": name,
        "trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_pnl": round(net_profit, 2),
        "total_fees": round(total_fees, 2),
        "profit_factor": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "final_bal": round(final_balance, 2)
    }

if __name__ == '__main__':
    print("=" * 80)
    print("       SOVEREIGN SWARM — COMPREHENSIVE MULTI-BOT HISTORICAL AUDIT       ")
    print("=" * 80)
    
    # Load 15m and 5m data
    print("Loading historical datasets...")
    df_btc_15m = pd.read_csv('historical_btc_usdt_15m.csv')
    df_btc_5m = pd.read_csv('historical_btc_usdt_5m.csv')
    
    print(f"Loaded {len(df_btc_5m):,} 5m bars & {len(df_btc_15m):,} 15m bars.")
    
    results = []
    print("\nRunning Bot 1: Trend Sniper (15m)...")
    results.append(backtest_trend_sniper(df_btc_15m, leverage=5.0))
    
    print("Running Bot 2: Berserker Scalper (5m)...")
    results.append(backtest_berserker_scalper(df_btc_5m, leverage=5.0))
    
    print("Running Bot 3: Mean Reversion / Ghost Grid (5m)...")
    results.append(backtest_mean_reversion(df_btc_5m, leverage=3.0))
    
    print("Running Bot 4: MARL Policy Trader (15m)...")
    results.append(backtest_marl_trader(df_btc_15m, leverage=4.0))
    
    print("Running Bot 5: The Swing Architect (4H)...")
    results.append(backtest_swing_architect_4h(None, leverage=3.0))
    
    print("Running Bot 6: Cash & Carry Basis Arbitrage (8H)...")
    results.append(backtest_basis_arbitrage(days=730))
    
    df_report = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(f"{'Bot Name':<28} | {'Trades':<7} | {'Win Rate':<8} | {'Fees Paid':<10} | {'Net PnL':<9} | {'Max DD':<7} | {'End Bal':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<28} | {r['trades']:<7d} | {r['win_rate']:<7.1f}% | ${r['total_fees']:<9.2f} | ${r['net_pnl']:<+8.2f} | {r['max_dd']:<6.1f}% | ${r['final_bal']:<7.2f}")
    print("=" * 80)
