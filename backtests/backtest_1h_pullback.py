import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def fetch_1h_data(symbol, start_year=2024):
    clean = symbol.replace('/', '')
    filepath = f"data/historical/{clean}_1h_full.csv"
    import os
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    
    ex = ccxt.binance({'enableRateLimit': True})
    start_ts = int(datetime(start_year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    all_bars = []
    since = start_ts
    while True:
        try:
            batch = ex.fetch_ohlcv(symbol, '1h', since=since, limit=1000)
            if not batch: break
            all_bars.extend(batch)
            if len(batch) < 1000: break
            since = batch[-1][0] + (3600 * 1000)
            time.sleep(0.05)
        except Exception as e:
            print(f"Error {symbol}: {e}")
            break
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.to_csv(filepath, index=False)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def backtest_1h_pullback(df, symbol="BTC/USDT", leverage=3.0, fee_rate=0.0006):
    # Indicators
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 4H-equivalent / Daily Trend: 200 EMA and 50 EMA on 1H (200 hours ~ 8 days, 50 hours ~ 2 days)
    # To mimic 4H EMA 50/200, use 200 and 800 on 1H
    df['ema_fast'] = close.ewm(span=21, adjust=False).mean()
    df['ema_mid'] = close.ewm(span=50, adjust=False).mean()
    df['ema_macro'] = close.ewm(span=200, adjust=False).mean()
    
    # ATR(14)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100.0 - (100.0 / (1.0 + rs))
    
    # Simulation
    balance = 100.0
    initial_balance = 100.0
    active_pos = None
    trades = []
    
    for i in range(250, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        curr_p = row['close']
        high_p = row['high']
        low_p = row['low']
        dt = row['dt']
        
        # Manage active position
        if active_pos is not None:
            side = active_pos['side']
            entry_p = active_pos['entry_price']
            sl = active_pos['sl']
            tp = active_pos['tp']
            
            # Check exit
            closed = False
            exit_p = None
            reason = None
            
            if side == 'LONG':
                if low_p <= sl:
                    exit_p = sl
                    reason = 'SL'
                    closed = True
                elif high_p >= tp:
                    exit_p = tp
                    reason = 'TP'
                    closed = True
                # Trailing stop to breakeven after +1.5R
                elif (curr_p - entry_p) >= (1.5 * active_pos['risk_unit']) and sl < entry_p:
                    active_pos['sl'] = entry_p + (0.05 * active_pos['risk_unit'])
            else:
                if high_p >= sl:
                    exit_p = sl
                    reason = 'SL'
                    closed = True
                elif low_p <= tp:
                    exit_p = tp
                    reason = 'TP'
                    closed = True
                elif (entry_p - curr_p) >= (1.5 * active_pos['risk_unit']) and sl > entry_p:
                    active_pos['sl'] = entry_p - (0.05 * active_pos['risk_unit'])
                    
            if closed:
                qty = active_pos['qty']
                notional = qty * exit_p
                exit_fee = notional * fee_rate
                gross_pnl = (exit_p - entry_p) * qty if side == 'LONG' else (entry_p - exit_p) * qty
                net_pnl = gross_pnl - active_pos['entry_fee'] - exit_fee
                balance += net_pnl
                trades.append({
                    'symbol': symbol,
                    'side': side,
                    'entry_time': active_pos['dt'],
                    'exit_time': dt,
                    'entry_p': entry_p,
                    'exit_p': exit_p,
                    'pnl': net_pnl,
                    'reason': reason
                })
                active_pos = None
                continue
                
        # Look for Entry if no position
        if active_pos is None and balance > 10.0:
            # Bullish Trend: Price > EMA 200 and EMA 50 > EMA 200
            is_bull_trend = (row['close'] > row['ema_macro']) and (row['ema_mid'] > row['ema_macro'])
            # Pullback to EMA 21 or EMA 50 with RSI recovering from oversold (<45 crossing up)
            pullback_long = (low_p <= row['ema_fast'] * 1.002) and (prev_row['rsi'] < 45) and (row['rsi'] >= 45) and (row['close'] > row['open'])
            
            if is_bull_trend and pullback_long:
                entry_p = curr_p
                sl_dist = max(1.5 * row['atr'], entry_p * 0.015)
                sl = entry_p - sl_dist
                tp = entry_p + (2.5 * sl_dist)
                risk_unit = entry_p - sl
                
                margin = min(20.0, balance * 0.20)
                notional = margin * leverage
                qty = notional / entry_p
                entry_fee = notional * fee_rate
                
                active_pos = {
                    'side': 'LONG',
                    'dt': dt,
                    'entry_price': entry_p,
                    'qty': qty,
                    'sl': sl,
                    'tp': tp,
                    'risk_unit': risk_unit,
                    'entry_fee': entry_fee
                }
                
    df_res = pd.DataFrame(trades)
    if df_res.empty:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0}
    wins = len(df_res[df_res['pnl'] > 0])
    wr = (wins / len(df_res)) * 100.0
    net_pnl = df_res['pnl'].sum()
    return {
        'symbol': symbol,
        'trades': len(df_res),
        'wins': wins,
        'win_rate': wr,
        'pnl': net_pnl,
        'final_balance': balance
    }

if __name__ == '__main__':
    for sym in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'LINK/USDT']:
        df = fetch_1h_data(sym, start_year=2024)
        res = backtest_1h_pullback(df, symbol=sym)
        print(f"{sym:10} | Bars: {len(df):5d} | Trades: {res['trades']:3d} | WR: {res['win_rate']:5.1f}% | Net PnL: ${res['pnl']:+8.2f} (End Bal: ${res['final_balance']:.2f})")
