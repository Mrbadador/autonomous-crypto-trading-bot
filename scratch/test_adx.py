import pandas as pd
import numpy as np

def calculate_adx_fallback(df, length=14):
    df = df.copy()
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    # Fill first NaN of tr
    df['tr'] = df['tr'].fillna(df['high'] - df['low'])
    
    # Welles Wilder's Smoothing Average for TR, +DM, -DM
    tr_smooth = df['tr'].rolling(window=length).sum()
    plus_dm_smooth = df['plus_dm'].rolling(window=length).sum()
    minus_dm_smooth = df['minus_dm'].rolling(window=length).sum()
    
    tr_smooth_val = tr_smooth.values.copy()
    plus_dm_smooth_val = plus_dm_smooth.values.copy()
    minus_dm_smooth_val = minus_dm_smooth.values.copy()
    
    tr_val = df['tr'].values
    plus_dm_val = df['plus_dm'].values
    minus_dm_val = df['minus_dm'].values
    
    for i in range(length, len(df)):
        tr_smooth_val[i] = tr_smooth_val[i-1] - (tr_smooth_val[i-1] / length) + tr_val[i]
        plus_dm_smooth_val[i] = plus_dm_smooth_val[i-1] - (plus_dm_smooth_val[i-1] / length) + plus_dm_val[i]
        minus_dm_smooth_val[i] = minus_dm_smooth_val[i-1] - (minus_dm_smooth_val[i-1] / length) + minus_dm_val[i]
        
    df['tr_smooth'] = tr_smooth_val
    df['plus_dm_smooth'] = plus_dm_smooth_val
    df['minus_dm_smooth'] = minus_dm_smooth_val
    
    plus_di = 100.0 * (df['plus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    minus_di = 100.0 * (df['minus_dm_smooth'] / (df['tr_smooth'] + 1e-9))
    
    dx = 100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    
    adx_series = dx.rolling(window=length).mean()
    adx_series_val = adx_series.values.copy()
    dx_val = dx.values
    
    for i in range(2 * length - 1, len(df)):
        if not np.isnan(adx_series_val[i-1]):
            adx_series_val[i] = adx_series_val[i-1] - (adx_series_val[i-1] / length) + (dx_val[i] / length)
            
    df['adx'] = adx_series_val
    return df

# Create dummy data
np.random.seed(42)
n = 100
high = 100.0 + np.cumsum(np.random.normal(0, 1, n))
low = high - 2.0 - np.random.exponential(1, n)
close = low + np.random.uniform(0, 2, n)
df = pd.DataFrame({'high': high, 'low': low, 'close': close})

res = calculate_adx_fallback(df)
print("ADX slice:")
print(res['adx'].iloc[20:40])
