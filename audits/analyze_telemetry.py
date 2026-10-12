import sqlite3
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_market_regime_data(db_path, limit=500):
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT timestamp, regime, hurst, adx FROM market_regime ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching market_regime: {e}")
        return pd.DataFrame()

def get_microstructure_data(db_path, limit=500):
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT timestamp, symbol, liquidity_imbalance as OFI, vacuum_detected FROM oracle_microstructure ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching oracle_microstructure: {e}")
        return pd.DataFrame()

def get_oracle_state_data(db_path, limit=500):
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT timestamp, symbol, volatility, trend_strength, hurst, apen FROM oracle_state ORDER BY timestamp DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching oracle_state: {e}")
        return pd.DataFrame()

def analyze_data():
    print("--- Market Regime Analysis (trading_data.db) ---")
    df_regime = get_market_regime_data("trading_data.db", 500)
    if not df_regime.empty:
        print(f"Total records fetched: {len(df_regime)}")
        print("\nRegime Distribution:")
        print(df_regime['regime'].value_counts())
        
        print("\nStats by Regime:")
        stats = df_regime.groupby('regime')[['hurst', 'adx']].agg(['mean', 'std', 'min', 'max'])
        print(stats)
        
        print("\nRecent 5 entries:")
        print(df_regime.head(5))
    
    print("\n--- Microstructure Analysis (trading_ledger.db) ---")
    df_micro = get_microstructure_data("trading_ledger.db", 500)
    if not df_micro.empty:
        print(f"Total records fetched: {len(df_micro)}")
        print("\nOFI (Order Flow Imbalance) Stats:")
        print(df_micro['OFI'].describe())
        print("\nVacuum Detected counts:")
        print(df_micro['vacuum_detected'].value_counts())
        
    print("\n--- Oracle State Analysis (trading_ledger.db) ---")
    df_state = get_oracle_state_data("trading_ledger.db", 500)
    if not df_state.empty:
        print(f"Total records fetched: {len(df_state)}")
        print("\nHurst & Volatility Stats:")
        print(df_state[['hurst', 'volatility', 'trend_strength', 'apen']].describe())
        
if __name__ == "__main__":
    analyze_data()
