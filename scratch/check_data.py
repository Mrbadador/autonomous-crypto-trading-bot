import pandas as pd
import indicators
import config
import os

config.TIMEFRAME = '1h'
config.OHLCV_LIMIT = 1000

df = pd.read_csv('data/historical/BTCUSDT_1h.csv')
processed = indicators.compute_indicators(df)

# Mock 4h EMA
df['ts_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
h4_df = df.set_index('ts_dt').resample('4h', label='left', closed='left').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
})
h4_df['ema200'] = h4_df['close'].ewm(span=200, adjust=False).mean()

h1_df = processed.copy()
h1_df['ts_dt'] = pd.to_datetime(h1_df['timestamp'], unit='ms')
h1_df = pd.merge_asof(h1_df.sort_values('ts_dt'), h4_df[['ema200']].rename(columns={'ema200': 'ema200_4h'}), left_on='ts_dt', right_index=True)

print("--- Potential LONG entries ---")
# Simplified Long: ADX > 20, ADX rising (adx > adx_avg), Price > EMA200_1h, Price > EMA200_4h
long_mask = (h1_df['adx'] > 20) & (h1_df['adx'] > h1_df['adx_avg']) & (h1_df['close'] > h1_df['ema200']) & (h1_df['close'] > h1_df['ema200_4h'])
print(f"Potential Long Trades: {long_mask.sum()}")

print("\n--- Potential SHORT entries ---")
# Simplified Short: ADX > 20, ADX rising, Price < EMA200_1h, Price < EMA200_4h
short_mask = (h1_df['adx'] > 20) & (h1_df['adx'] > h1_df['adx_avg']) & (h1_df['close'] < h1_df['ema200']) & (h1_df['close'] < h1_df['ema200_4h'])
print(f"Potential Short Trades: {short_mask.sum()}")
