"""
test_mean_reversion_maker.py — Lightning-fast Parameter Sweep for Mean Reversion Maker (Longs Only)
"""
import pandas as pd
import numpy as np

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

print("Loading historical data once...")
import os
csv_path = "data/historical/BTCUSDT_5m.csv" if os.path.exists("data/historical/BTCUSDT_5m.csv") else ("data/historical/historical_btc_usdt_5m.csv" if os.path.exists("data/historical/historical_btc_usdt_5m.csv") else "historical_btc_usdt_5m.csv")
df = pd.read_csv(csv_path)
c, h, l = df['close'], df['high'], df['low']
df['rsi'] = compute_rsi(c, 14)
df['atr'] = compute_atr(h, l, c, 14)
df['adx'] = compute_adx(h, l, c, 14)

# 2-hour VWAP & 2-hour Bollinger Bands
cum_pv = (c * df['volume']).rolling(24).sum()
cum_v = df['volume'].rolling(24).sum()
df['vwap'] = cum_pv / (cum_v + 1e-9)
df['rolling_std'] = c.rolling(24).std()
df['lower_band_15'] = df['vwap'] - (1.5 * df['rolling_std'])
df['lower_band_20'] = df['vwap'] - (2.0 * df['rolling_std'])

# Macro trend filter (4H ADX & 200 EMA proxy)
df['ema_200'] = c.ewm(span=1152, adjust=False).mean()
df['adx_4h'] = df['adx'].rolling(48).mean()
print("Data prepared. Running sweeps...")

def evaluate_grid(df_data, tp_pct, sl_pct, rsi_entry, band_type='lower_band_15', maker_fee=0.0002, taker_fee=0.0006):
    balance = 100.0
    trades = []
    pos = None
    leverage = 3.0
    
    closes = df_data['close'].values
    highs = df_data['high'].values
    lows = df_data['low'].values
    rsis = df_data['rsi'].values
    bands = df_data[band_type].values
    adx_4hs = df_data['adx_4h'].values
    adxs = df_data['adx'].values
    ema_200s = df_data['ema_200'].values
    
    n = len(closes)
    for i in range(1200, n):
        curr_p = closes[i]
        high_p = highs[i]
        low_p = lows[i]
        
        if pos:
            entry_p = pos['entry_p']
            sl = pos['sl']
            tp = pos['tp']
            qty = pos['qty']
            
            closed = False
            exit_p = None
            fee_exit = 0.0
            
            if low_p <= sl:
                exit_p = sl
                fee_exit = qty * exit_p * taker_fee
                closed = True
            elif high_p >= tp:
                exit_p = tp
                fee_exit = qty * exit_p * maker_fee
                closed = True
                
            if closed:
                gross = (exit_p - entry_p) * qty
                total_fees = pos['fee_entry'] + fee_exit
                net = gross - total_fees
                balance += net
                trades.append({'pnl': net, 'fees': total_fees})
                pos = None
                continue
                
        is_chop = (adx_4hs[i] < 23.0) or (adxs[i] < 20.0)
        not_in_crash = curr_p > (ema_200s[i] * 0.95)
        
        if not pos and balance > 10.0 and is_chop and not_in_crash:
            if (rsis[i] < rsi_entry) and (curr_p < bands[i]):
                entry_p = curr_p
                tp = entry_p * (1.0 + tp_pct)
                sl = entry_p * (1.0 - sl_pct)
                
                margin = min(20.0, balance * 0.15)
                notional = margin * leverage
                qty = notional / entry_p
                fee_entry = notional * maker_fee
                pos = {'entry_p': entry_p, 'qty': qty, 'sl': sl, 'tp': tp, 'fee_entry': fee_entry}

    total_trades = len(trades)
    if total_trades == 0:
        return 0, 0.0, 0.0, 0.0
    wins = [t for t in trades if t['pnl'] > 0]
    wr = len(wins) / total_trades * 100.0
    net_pnl = balance - 100.0
    fees = sum(t['fees'] for t in trades)
    return total_trades, wr, net_pnl, fees

results = []
for band in ['lower_band_15', 'lower_band_20']:
    for rsi in [25.0, 28.0, 32.0]:
        for tp in [0.008, 0.012, 0.016, 0.020]:
            for sl in [0.015, 0.020, 0.025]:
                trades, wr, pnl, fees = evaluate_grid(df, tp, sl, rsi, band)
                results.append((pnl, wr, trades, fees, band, rsi, tp, sl))

results.sort(key=lambda x: x[0], reverse=True)
print("\n--- TOP 10 PROFITABLE CONFIGURATIONS ---")
print(f"{'Band':16} | {'RSI':4} | {'TP':5} | {'SL':5} | {'Trades':6} | {'Win Rate':8} | {'Net PnL':10} | {'Fees':6}")
print("-" * 75)
for r in results[:10]:
    pnl, wr, trades, fees, band, rsi, tp, sl = r
    print(f"{band:16} | {rsi:4.0f} | {tp*100:4.1f}% | {sl*100:4.1f}% | {trades:6d} | {wr:7.1f}% | ${pnl:+8.2f} | ${fees:5.2f}")
