import sqlite3
import pandas as pd
import numpy as np

def analyze():
    print("Extracting from trading_ledger.db...")
    conn = sqlite3.connect("trading_ledger.db")
    
    # Get Oracle State (Hurst, Trend Strength, etc)
    state_df = pd.read_sql_query("""
        SELECT timestamp, symbol, hurst, trend_strength, volatility
        FROM oracle_state
        ORDER BY timestamp DESC
        LIMIT 2000
    """, conn)
    
    # Get Microstructure (OFI, Vacuum)
    micro_df = pd.read_sql_query("""
        SELECT timestamp, symbol, liquidity_imbalance as ofi, vacuum_detected
        FROM oracle_microstructure
        ORDER BY timestamp DESC
        LIMIT 2000
    """, conn)
    
    conn.close()

    conn2 = sqlite3.connect("trading_data.db")
    regime_df = pd.read_sql_query("""
        SELECT timestamp, regime, hurst as regime_hurst, adx
        FROM market_regime
        ORDER BY timestamp DESC
        LIMIT 2000
    """, conn2)
    conn2.close()

    print("\n--- Oracle State Stats (Last 2000) ---")
    if not state_df.empty:
        print(state_df[['hurst', 'trend_strength', 'volatility']].describe())
    else:
        print("No oracle_state data")

    print("\n--- Microstructure Stats (Last 2000) ---")
    if not micro_df.empty:
        print(micro_df[['ofi', 'vacuum_detected']].describe())
        # Let's do a rolling average of OFI
        micro_df['ofi_rolling_10'] = micro_df['ofi'].rolling(10).mean()
        print("\nOFI Rolling Average (10 periods) Stats:")
        print(micro_df['ofi_rolling_10'].describe())
    else:
        print("No microstructure data")

    print("\n--- Regime Stats (Last 2000) ---")
    if not regime_df.empty:
        print(regime_df[['regime_hurst', 'adx']].describe())
    else:
        print("No market_regime data")

    # Let's join on timestamp and symbol if possible, or just look at latest
    if not state_df.empty and not micro_df.empty:
        merged = pd.merge(state_df, micro_df, on=['timestamp', 'symbol'], how='inner')
        if not merged.empty:
            print(f"\n--- Correlation Analysis (N={len(merged)}) ---")
            print(merged[['hurst', 'trend_strength', 'volatility', 'ofi']].corr())
            
            # Slice the last 50 periods
            recent = merged.head(50)
            print("\nRecent 50 periods means:")
            print(recent[['hurst', 'trend_strength', 'volatility', 'ofi']].mean())

analyze()
