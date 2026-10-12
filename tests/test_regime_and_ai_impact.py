import pandas as pd
import numpy as np

# Load 15m BTC data (70,080 bars = 2 full years)
df = pd.read_csv('historical_btc_usdt_15m.csv')
c, h, l, v = df['close'], df['high'], df['low'], df['volume']

# Technical Indicators
df['ema_20'] = c.ewm(span=20, adjust=False).mean()
df['ema_50'] = c.ewm(span=50, adjust=False).mean()
df['ema_200'] = c.ewm(span=200, adjust=False).mean()

# ATR
prev_c = c.shift(1)
tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# RSI
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
df['rsi'] = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))

# ADX
up = h - h.shift(1)
dn = l.shift(1) - l
pdm = np.where((up > dn) & (up > 0), up, 0.0)
mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
pdi = 100.0 * pd.Series(pdm).rolling(14).mean() / (df['atr'] + 1e-9)
mdi = 100.0 * pd.Series(mdm).rolling(14).mean() / (df['atr'] + 1e-9)
df['adx'] = (100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)).rolling(14).mean()

# Volume EMA & Ratio
df['vol_ema'] = v.ewm(span=20, adjust=False).mean()
df['vol_ratio'] = v / (df['vol_ema'] + 1e-9)

# Macro Trend (Daily equivalent on 15m = 96 bars/day -> 96 * 50 = 4800, or simple EMA 800)
df['ema_macro'] = c.ewm(span=800, adjust=False).mean()

def run_simulation(name, use_regime_filter=False, use_ai_filter=False, fee_rate=0.0006):
    balance = 100.0
    trades = []
    pos = None
    leverage = 4.0
    
    for i in range(250, len(df)):
        r = df.iloc[i]
        curr_p, hi, lo, atr = r['close'], r['high'], r['low'], r['atr']
        
        # Position Management
        if pos:
            side = pos['side']
            sl, tp = pos['sl'], pos['tp']
            closed = False
            exit_p = None
            
            if side == 'LONG':
                if lo <= sl: exit_p, closed = sl, True
                elif hi >= tp: exit_p, closed = tp, True
                # Trailing stop to breakeven after +1.5R
                elif (curr_p - pos['p']) >= (1.5 * pos['risk_unit']) and sl < pos['p']:
                    pos['sl'] = pos['p'] + (0.05 * pos['risk_unit'])
            else:
                if hi >= sl: exit_p, closed = sl, True
                elif lo <= tp: exit_p, closed = tp, True
                elif (pos['p'] - curr_p) >= (1.5 * pos['risk_unit']) and sl > pos['p']:
                    pos['sl'] = pos['p'] - (0.05 * pos['risk_unit'])
                    
            if closed:
                qty = pos['qty']
                gross = (exit_p - pos['p']) * qty if side == 'LONG' else (pos['p'] - exit_p) * qty
                fee = (pos['p'] * qty * fee_rate) + (exit_p * qty * fee_rate)
                net = gross - fee
                balance += net
                trades.append({'net': net, 'gross': gross, 'fee': fee, 'side': side})
                pos = None
                continue
                
        # Entry Logic
        if not pos and balance > 10.0:
            # 1. Base Technical Signal: EMA 20/50 Cross
            base_long = (r['ema_20'] > r['ema_50']) and (r['close'] > r['ema_20'])
            base_short = (r['ema_20'] < r['ema_50']) and (r['close'] < r['ema_20'])
            
            # 2. Regime Filter Gate:
            # If enabled, only trade if market is verified TRENDING (ADX > 22, EMA stack aligned with EMA200)
            if use_regime_filter:
                regime_long_ok = (r['adx'] > 22.0) and (r['ema_50'] > r['ema_200']) and (r['close'] > r['ema_200'])
                regime_short_ok = (r['adx'] > 22.0) and (r['ema_50'] < r['ema_200']) and (r['close'] < r['ema_200'])
                base_long = base_long and regime_long_ok
                base_short = base_short and regime_short_ok
                
            # 3. AI Conviction Filter Gate (Boardroom & Macro Sentinel proxy):
            # Requires Macro Alignment (SPY/Macro trend) + Volume Confirmation (>1.3x) + Non-exhausted RSI (45-65)
            if use_ai_filter:
                ai_long_ok = (r['close'] > r['ema_macro']) and (r['vol_ratio'] > 1.30) and (45.0 <= r['rsi'] <= 65.0)
                ai_short_ok = (r['close'] < r['ema_macro']) and (r['vol_ratio'] > 1.30) and (35.0 <= r['rsi'] <= 55.0)
                base_long = base_long and ai_long_ok
                base_short = base_short and ai_short_ok
                
            if base_long or base_short:
                side = 'LONG' if base_long else 'SHORT'
                p = curr_p
                sl_dist = 2.0 * atr
                sl = p - sl_dist if side == 'LONG' else p + sl_dist
                tp = p + (3.0 * sl_dist) if side == 'LONG' else p - (3.0 * sl_dist)
                risk_unit = sl_dist
                
                margin = min(15.0, balance * 0.15)
                qty = (margin * leverage) / p
                pos = {'side': side, 'p': p, 'sl': sl, 'tp': tp, 'risk_unit': risk_unit, 'qty': qty}
                
    total_trades = len(trades)
    if total_trades == 0:
        return {'name': name, 'trades': 0, 'wr': 0, 'fees': 0, 'net': 0, 'bal': balance}
    wins = [t for t in trades if t['net'] > 0]
    wr = (len(wins) / total_trades) * 100.0
    tot_fees = sum(t['fee'] for t in trades)
    net_pnl = balance - 100.0
    
    # Drawdown
    equity = [100.0]
    curr = 100.0
    peak = 100.0
    max_dd = 0.0
    for t in trades:
        curr += t['net']
        peak = max(peak, curr)
        dd = ((peak - curr) / peak) * 100.0 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        
    return {
        'name': name,
        'trades': total_trades,
        'wr': round(wr, 1),
        'fees': round(tot_fees, 2),
        'net': round(net_pnl, 2),
        'max_dd': round(max_dd, 1),
        'bal': round(balance, 2)
    }

print("=" * 80)
print("     STEP-BY-STEP IMPACT: REGIME FILTER vs AI CONVICTION vs MAKER FEES")
print("=" * 80)

cases = [
    ("1. Raw Technical Only (Taker 0.06%)", False, False, 0.0006),
    ("2. + Regime Filter (ADX>22, Taker 0.06%)", True, False, 0.0006),
    ("3. + Regime + AI Boardroom (Taker 0.06%)", True, True, 0.0006),
    ("4. + Regime + AI + Maker Limit (0.02%)", True, True, 0.0002),
]

for name, use_reg, use_ai, fee in cases:
    res = run_simulation(name, use_regime_filter=use_reg, use_ai_filter=use_ai, fee_rate=fee)
    print(f"{res['name']:<42} | Trades: {res['trades']:<5d} | WR: {res['wr']:<5.1f}% | Fees: ${res['fees']:<7.2f} | Net: ${res['net']:<+7.2f} | DD: {res['max_dd']:<5.1f}% | Bal: ${res['bal']:<6.2f}")
print("=" * 80)
