import sqlite3
import datetime

def inject_extreme_panic():
    try:
        with sqlite3.connect("trading_ledger.db", timeout=15.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            
            # Inject Extreme Fear into market_context
            conn.execute("""
                INSERT INTO market_context 
                (timestamp, fear_greed_index, volatility_state, trend_strength, vwap, rsi_14, signal_readiness, whale_threat) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.datetime.utcnow().isoformat(), 15.0, 1.5, -0.5, 60000, 30, 0.5, 0.9))
            
            # Inject Severe Crash into macro_logs
            conn.execute("""
                INSERT INTO macro_logs
                (timestamp, symbol, spy_health, dxy_momentum, source)
                VALUES (?, ?, ?, ?, ?)
            """, (datetime.datetime.utcnow().isoformat(), "BTC/USDT", -0.8, 0.1, "oracle"))
            
            conn.commit()
            print("Successfully injected Extreme Panic (Fear & Greed = 15.0, SPY Health = -0.8) into DB.")
    except Exception as e:
        print(f"Failed to inject: {e}")

if __name__ == "__main__":
    inject_extreme_panic()
