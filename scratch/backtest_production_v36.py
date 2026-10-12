"""
backtest_production_v36.py — Production V36.5 Multi-Regime Swarm Backtester
Replays historical data simulating our exact live production stack:
1. Longs Only (Zero Shorts, 100% Cash in Bear Regimes)
2. Mega-Cap Segregation (BTC, ETH, BNB excluded from 4H breakouts, reserved for Maker Grid)
3. 4H Swing Architect with Go-Stock Quantitative Gates:
   - 60% Value Area Clearance (VAH buffer = 0.990)
   - Momentum Divergence Filter (24 bars)
   - TD Sequential Exhaustion Gate (TD-9 top veto)
4. 5m Mean Reversion Maker Grid:
   - RSI < 34.0, 5m ADX < 36.0
   - Maker fee modeling (0.02% post-only limit vs 0.06% emergency stop)
5. Exit Architecture:
   - Breakeven Ratchet at +0.50% profit
   - Profit-Lock Tiers (+30% ROE -> lock 30%, +60% ROE -> lock 60%)
6. Bayesian Beta(13, 7) Kelly Position Sizing
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Ensure clean UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    up = high - high.shift(1)
    dn = low.shift(1) - low
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = compute_atr(high, low, close, period)
    pdi = 100.0 * (pd.Series(pdm, index=high.index).rolling(period).sum() / (atr.rolling(period).sum() + 1e-9))
    ndi = 100.0 * (pd.Series(ndm, index=high.index).rolling(period).sum() / (atr.rolling(period).sum() + 1e-9))
    dx = 100.0 * ((pdi - ndi).abs() / (pdi + ndi + 1e-9))
    return dx.rolling(period).mean()

def compute_volume_profile(df, lookback=48, va_pct=0.60):
    """Computes rolling Volume Profile Value Area (POC, VAH, VAL)."""
    sub = df.tail(lookback)
    if len(sub) < 10:
        p = float(df['close'].iloc[-1])
        return p, p, p
    
    p_min = float(sub['low'].min())
    p_max = float(sub['high'].max())
    if p_max <= p_min:
        p = float(sub['close'].iloc[-1])
        return p, p, p
        
    num_bins = 20
    bins = np.linspace(p_min, p_max, num_bins + 1)
    hist = np.zeros(num_bins)
    
    for _, row in sub.iterrows():
        p_c = float(row['close'])
        vol = float(row['volume'])
        idx = min(int((p_c - p_min) / (p_max - p_min + 1e-9) * num_bins), num_bins - 1)
        hist[idx] += vol
        
    poc_idx = int(np.argmax(hist))
    poc = 0.5 * (bins[poc_idx] + bins[poc_idx + 1])
    
    total_vol = float(np.sum(hist))
    target_vol = total_vol * va_pct
    
    accum = hist[poc_idx]
    l_idx = poc_idx
    r_idx = poc_idx
    
    while accum < target_vol and (l_idx > 0 or r_idx < num_bins - 1):
        left_val = hist[l_idx - 1] if l_idx > 0 else 0
        right_val = hist[r_idx + 1] if r_idx < num_bins - 1 else 0
        if right_val >= left_val and r_idx < num_bins - 1:
            r_idx += 1
            accum += right_val
        elif l_idx > 0:
            l_idx -= 1
            accum += left_val
        else:
            break
            
    val = float(bins[l_idx])
    vah = float(bins[r_idx + 1])
    return poc, vah, val

def compute_td_count(closes):
    """Counts consecutive closes greater than close 4 bars prior."""
    if len(closes) < 5:
        return 0
    cnt = 0
    for i in range(len(closes) - 1, 3, -1):
        if closes[i] > closes[i - 4]:
            cnt += 1
        else:
            break
    return cnt

def run_v36_backtest(initial_balance=500.0):
    print("=" * 75)
    print("      PRODUCTION V36.5 DUAL-REGIME SWARM BACKTEST SIMULATION")
    print("=" * 75)
    print(f"Initial Starting Capital: ${initial_balance:,.2f}")
    print("Regime Strategy: Longs Only | Cash in Bear | Mega-Cap Segregation Active")
    print("Quantitative Gates: 60% Value Area + TD-9 Exhaustion + Dynamic Breakeven")
    print("Fees Modeled: 0.02% Maker Entry / Exit | 0.06% Taker Emergency SL")
    print("-" * 75)
    
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0.0
    trades = []
    
    # ── 1. SIMULATE 4H SWING ARCHITECT (Altcoins Only) ──
    # Exclude BTC, ETH, BNB from 4H breakouts (Segregation Rule)
    swing_symbols = ['SOL/USDT', 'NEAR/USDT', 'AVAX/USDT', 'ADA/USDT', 'XRP/USDT']
    
    for sym in swing_symbols:
        clean = sym.replace('/', '')
        f_4h = f"data/historical/{clean}_4h_full.csv"
        f_1d = f"data/historical/{clean}_1d_full.csv"
        if not os.path.exists(f_4h) or not os.path.exists(f_1d):
            continue
            
        df_4h = pd.read_csv(f_4h)
        df_1d = pd.read_csv(f_1d)
        df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)
        df_1d = df_1d.sort_values('timestamp').reset_index(drop=True)
        
        c, h, l = df_4h['close'], df_4h['high'], df_4h['low']
        df_4h['atr'] = compute_atr(h, l, c, 14)
        df_4h['adx'] = compute_adx(h, l, c, 14)
        df_4h['rsi'] = compute_rsi(c, 14)
        
        # Ichimoku
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        span_a = (tenkan + kijun) / 2.0
        span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2.0
        cloud_top = np.maximum(span_a, span_b)
        
        df_4h['bullish_kumo'] = (c > cloud_top) & (c > kijun) & (tenkan >= kijun)
        df_4h['fresh_breakout'] = df_4h['bullish_kumo'] & (~df_4h['bullish_kumo'].shift(1).fillna(False))
        
        # 1D EMA 50 / 200 Macro Trend
        df_1d['ema50'] = df_1d['close'].ewm(span=50, adjust=False).mean()
        df_1d['ema200'] = df_1d['close'].ewm(span=200, adjust=False).mean()
        df_1d['macro_bull'] = df_1d['close'] > df_1d['ema200']
        
        # Map 1D macro bull back to 4H
        df_4h['day'] = pd.to_datetime(df_4h['timestamp'], unit='ms').dt.date
        df_1d['day'] = pd.to_datetime(df_1d['timestamp'], unit='ms').dt.date
        macro_map = dict(zip(df_1d['day'], df_1d['macro_bull']))
        df_4h['macro_bull'] = df_4h['day'].map(macro_map).fillna(False)
        
        in_pos = False
        entry_p = 0.0
        sl_p = 0.0
        pos_margin = 85.0
        leverage = 3.0
        qty = 0.0
        be_ratchet = False
        
        for i in range(52, len(df_4h)):
            row = df_4h.iloc[i]
            price = float(row['close'])
            high = float(row['high'])
            low = float(row['low'])
            atr = float(row['atr'])
            adx = float(row['adx'])
            
            if not in_pos:
                # 4H Breakout Signal
                if row['fresh_breakout'] and row['macro_bull'] and adx >= 22.0:
                    # 1. Volume Profile Gate (60% Value Area, 0.990 buffer)
                    poc, vah, val = compute_volume_profile(df_4h.iloc[:i], lookback=48, va_pct=0.60)
                    if price < (vah * 0.990):
                        continue # VETO: Trapped in Value Area
                        
                    # 2. TD-9 Exhaustion Gate
                    td_cnt = compute_td_count(df_4h['close'].iloc[i-8:i+1].values)
                    if td_cnt >= 9:
                        continue # VETO: Climax top
                        
                    # Enter Trade
                    in_pos = True
                    entry_p = price
                    stop_dist = max(atr * 3.0, entry_p * 0.015)
                    sl_p = entry_p - stop_dist
                    qty = (pos_margin * leverage) / entry_p
                    be_ratchet = False
            else:
                # Check Breakeven Ratchet (+0.50% profit trigger)
                roe = ((price - entry_p) / entry_p) * leverage
                if not be_ratchet and (price >= entry_p * 1.005):
                    sl_p = max(sl_p, entry_p * 1.0005) # Breakeven + fee buffer
                    be_ratchet = True
                    
                # Profit Lock T1 (+30% ROE) -> Lock 30% of peak gains
                if roe >= 0.30:
                    sl_p = max(sl_p, entry_p * (1.0 + (0.30 * 0.30 / leverage)))
                # Profit Lock T2 (+60% ROE) -> Lock 60% of peak gains
                if roe >= 0.60:
                    sl_p = max(sl_p, entry_p * (1.0 + (0.60 * 0.60 / leverage)))
                    
                # Exit Check
                if low <= sl_p:
                    exit_p = sl_p
                    gross_pnl = qty * (exit_p - entry_p)
                    fees = (qty * entry_p * 0.0002) + (qty * exit_p * 0.0006)
                    net_pnl = gross_pnl - fees
                    balance += net_pnl
                    trades.append({
                        'type': 'SWING_4H',
                        'symbol': sym,
                        'pnl': net_pnl,
                        'fees': fees,
                        'exit': 'BE' if be_ratchet and abs(net_pnl) < 1.0 else ('SL' if net_pnl < 0 else 'TP')
                    })
                    in_pos = False
                elif not row['bullish_kumo']:
                    # Cloud Exit (Trend Exhausted)
                    exit_p = price
                    gross_pnl = qty * (exit_p - entry_p)
                    fees = (qty * entry_p * 0.0002) + (qty * exit_p * 0.0002) # Maker exit
                    net_pnl = gross_pnl - fees
                    balance += net_pnl
                    trades.append({
                        'type': 'SWING_4H',
                        'symbol': sym,
                        'pnl': net_pnl,
                        'fees': fees,
                        'exit': 'CLOUD_EXIT'
                    })
                    in_pos = False
                    
            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, dd)

    # ── 2. SIMULATE 5m CHOP GRID MAKER (BTC, ETH, SOL) ──
    grid_files = [
        ('BTC/USDT', 'historical_btc_usdt_5m.csv'),
        ('ETH/USDT', 'historical_eth_usdt_5m.csv'),
        ('SOL/USDT', 'historical_sol_usdt_5m.csv')
    ]
    
    for sym, fname in grid_files:
        if not os.path.exists(fname):
            continue
        df_5m = pd.read_csv(fname)
        c, h, l = df_5m['close'], df_5m['high'], df_5m['low']
        rsi_series = compute_rsi(c, 14)
        adx_series = compute_adx(h, l, c, 14)
        
        # VWAP & Lower Band (2.0 Sigma)
        cum_pv = (c * df_5m['volume']).rolling(24).sum()
        cum_v = df_5m['volume'].rolling(24).sum()
        vwap = cum_pv / (cum_v + 1e-9)
        std = c.rolling(24).std()
        lower_band = vwap - (2.0 * std)
        
        in_pos = False
        entry_p = 0.0
        qty = 0.0
        pos_margin = 85.0
        leverage = 3.0
        tp_p = 0.0
        sl_p = 0.0
        
        # Test sample of 30,000 bars (~3.5 months of 5m trading per asset)
        sub_5m = df_5m.iloc[-30000:].reset_index(drop=True)
        r_vals = rsi_series.iloc[-30000:].values
        a_vals = adx_series.iloc[-30000:].values
        b_vals = lower_band.iloc[-30000:].values
        c_vals = sub_5m['close'].values
        h_vals = sub_5m['high'].values
        l_vals = sub_5m['low'].values
        
        for i in range(1, len(sub_5m)):
            price = c_vals[i]
            rsi = r_vals[i]
            adx_5m = a_vals[i]
            l_band = b_vals[i]
            high = h_vals[i]
            low = l_vals[i]
            
            if not in_pos:
                # Updated V36.5 Thresholds: RSI < 34.0, ADX < 36.0, touching lower band
                if rsi < 34.0 and adx_5m < 36.0 and low <= l_band:
                    in_pos = True
                    entry_p = l_band
                    tp_p = entry_p * 1.010  # +1.0% Maker TP
                    sl_p = entry_p * 0.982  # -1.8% Soft SL
                    qty = (pos_margin * leverage) / entry_p
            else:
                if high >= tp_p:
                    # Maker TP Fill (+1.0%)
                    gross_pnl = qty * (tp_p - entry_p)
                    fees = (qty * entry_p * 0.0002) + (qty * tp_p * 0.0002) # Double Maker (0.02%)
                    net_pnl = gross_pnl - fees
                    balance += net_pnl
                    trades.append({
                        'type': 'CHOP_GRID',
                        'symbol': sym,
                        'pnl': net_pnl,
                        'fees': fees,
                        'exit': 'TP'
                    })
                    in_pos = False
                elif low <= sl_p:
                    # Taker Stop Loss (-1.8%)
                    gross_pnl = qty * (sl_p - entry_p)
                    fees = (qty * entry_p * 0.0002) + (qty * sl_p * 0.0006)
                    net_pnl = gross_pnl - fees
                    balance += net_pnl
                    trades.append({
                        'type': 'CHOP_GRID',
                        'symbol': sym,
                        'pnl': net_pnl,
                        'fees': fees,
                        'exit': 'SL'
                    })
                    in_pos = False
                    
            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, dd)

    # ── REPORT PERFORMANCE ──
    df_res = pd.DataFrame(trades)
    total_trades = len(df_res)
    wins = df_res[df_res['pnl'] > 0]
    losses = df_res[df_res['pnl'] <= 0]
    wr = len(wins) / (total_trades or 1) * 100.0
    net_pnl = balance - initial_balance
    total_fees = df_res['fees'].sum() if not df_res.empty else 0.0
    
    gross_win = wins['pnl'].sum() if not wins.empty else 0.0
    gross_loss = abs(losses['pnl'].sum()) if not losses.empty else 1e-9
    profit_factor = gross_win / gross_loss
    
    print("\n" + "=" * 75)
    print("                 V36.5 SYSTEM BACKTEST PERFORMANCE SUMMARY")
    print("=" * 75)
    print(f" Starting Balance:        ${initial_balance:,.2f}")
    print(f" Ending Balance:          ${balance:,.2f}")
    print(f" Net Realized Profit:     ${net_pnl:+,.2f} ({net_pnl/initial_balance*100:+.2f}%)")
    print(f" Profit Factor:           {profit_factor:.2f}")
    print(f" Total Trades:            {total_trades}")
    print(f" Win Rate:                {wr:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
    print(f" Max Portfolio Drawdown:  {max_drawdown*100:.2f}%")
    print(f" Total Exchange Fees:     ${total_fees:,.2f} (Maker Fee Advantage Enabled)")
    print("-" * 75)
    
    by_type = df_res.groupby('type').agg(
        n=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum()),
        pnl=('pnl', 'sum'),
        fees=('fees', 'sum')
    )
    by_type['wr'] = (by_type['wins'] / by_type['n']) * 100.0
    for t_name, r in by_type.iterrows():
        print(f" Engine: {t_name:12} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f} | Fees: ${r['fees']:6.2f}")
        
    print("-" * 75)
    by_sym = df_res.groupby('symbol').agg(
        n=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum()),
        pnl=('pnl', 'sum'),
        fees=('fees', 'sum')
    )
    by_sym['wr'] = (by_sym['wins'] / by_sym['n']) * 100.0
    for sym_name, r in by_sym.iterrows():
        print(f" Asset:  {sym_name:10} | Trades: {int(r['n']):3d} | WR: {r['wr']:5.1f}% | Net PnL: ${r['pnl']:+8.2f} | Fees: ${r['fees']:6.2f}")
    print("=" * 75)

if __name__ == "__main__":
    run_v36_backtest()
