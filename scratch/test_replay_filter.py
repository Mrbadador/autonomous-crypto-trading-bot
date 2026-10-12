import pandas as pd
import os

# Mock config
class Config:
    IS_REPLAY = True
    REPLAY_START_DATE = "2023-04-01"
    REPLAY_END_DATE = "2023-07-31"
    OHLCV_LIMIT = 250

config = Config()

filepath = "data/historical/BTCUSDT_1h.csv"
df = pd.read_csv(filepath)
df.columns = [c.lower() for c in df.columns]
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df.set_index('timestamp', inplace=True)
df = df.sort_index()

print(f"Original: {df.index[0]} to {df.index[-1]} ({len(df)} rows)")

start = pd.to_datetime(config.REPLAY_START_DATE, utc=True)
end = pd.to_datetime(config.REPLAY_END_DATE, utc=True)

mask = df.index <= end
df_filtered = df[mask]

print(f"After End Filter: {df_filtered.index[0]} to {df_filtered.index[-1]} ({len(df_filtered)} rows)")

# get_indexer with bfill
start_idx_in_full = df_filtered.index.get_indexer([start], method='bfill')[0]
print(f"Start index in full: {start_idx_in_full} (Time: {df_filtered.index[start_idx_in_full]})")

if start_idx_in_full != -1:
    actual_start_idx = max(0, start_idx_in_full - config.OHLCV_LIMIT)
    df_final = df_filtered.iloc[actual_start_idx:]
    print(f"Final: {df_final.index[0]} to {df_final.index[-1]} ({len(df_final)} rows)")
    print(f"First 5 indices: {df_final.index[:5]}")
