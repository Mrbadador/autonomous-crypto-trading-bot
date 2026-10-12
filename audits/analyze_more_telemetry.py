import sqlite3
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def analyze_more_data():
    conn = sqlite3.connect("trading_data.db")
    
    print("--- Cycles Data ---")
    try:
        cycles_df = pd.read_sql_query("SELECT timestamp, symbol, regime, adx, atr_pct FROM cycles ORDER BY timestamp DESC LIMIT 100", conn)
        print(f"Cycles records: {len(cycles_df)}")
        if not cycles_df.empty:
            print("Regime distribution in cycles:")
            print(cycles_df['regime'].value_counts())
            print("Stats in cycles:")
            print(cycles_df[['adx', 'atr_pct']].describe())
    except Exception as e:
        print(e)

    print("\n--- Telemetry ---")
    try:
        tel_df = pd.read_sql_query("SELECT timestamp, symbol, reasoning, action FROM telemetry ORDER BY timestamp DESC LIMIT 20", conn)
        if not tel_df.empty:
            for idx, row in tel_df.iterrows():
                print(f"{row['timestamp']} | {row['symbol']} | {row['action']} | {row['reasoning']}")
    except Exception as e:
        print(e)
        
    print("\n--- System State ---")
    try:
        sys_state = pd.read_sql_query("SELECT key, value FROM system_state", conn)
        print(sys_state)
    except Exception as e:
        print(e)

    conn.close()

if __name__ == "__main__":
    analyze_more_data()
