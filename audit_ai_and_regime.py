import sqlite3
import json
import pandas as pd

print("=" * 60)
print("  PART 1: AI CONVICTION & BOARDROOM IMPACT ANALYSIS")
print("=" * 60)

for db_name in ['trading_data.db', 'trading_ledger.db']:
    try:
        conn = sqlite3.connect(db_name)
        tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"\nTables in {db_name}: {tables}")
    except Exception as e:
        print(f"Error {db_name}: {e}")

# Check boardroom deliberation logs
conn_data = sqlite3.connect('trading_data.db')
conn_ledger = sqlite3.connect('trading_ledger.db')

# Check trades with conviction scores or ai_rationale
try:
    cols = [c[1] for c in conn_data.execute("PRAGMA table_info(trades)").fetchall()]
    print(f"\nColumns in 'trades' table: {cols}")
    
    # Query trades where conviction score exists
    rows = conn_data.execute("SELECT symbol, side, pnl, action, notes FROM trades ORDER BY id DESC LIMIT 10").fetchall()
    print("\nRecent 10 trades from 'trades':")
    for r in rows:
        print(r)
except Exception as e:
    print(f"Error querying trades: {e}")

# Check ghost_trades (shadow signals evaluated without live execution)
try:
    ghost_cols = [c[1] for c in conn_data.execute("PRAGMA table_info(ghost_trades)").fetchall()]
    print(f"\nColumns in 'ghost_trades': {ghost_cols}")
    ghost_summary = conn_data.execute("""
        SELECT outcome, COUNT(*) as cnt, AVG(CAST(simulated_pnl AS REAL)) as avg_pnl 
        FROM ghost_trades 
        GROUP BY outcome
    """).fetchall()
    print("Ghost trades outcome summary:")
    for gs in ghost_summary:
        print(gs)
except Exception as e:
    print(f"Error querying ghost_trades: {e}")

# Check market_regime logs
try:
    reg_cols = [c[1] for c in conn_data.execute("PRAGMA table_info(market_regime)").fetchall()]
    print(f"\nColumns in 'market_regime': {reg_cols}")
    reg_summary = conn_data.execute("""
        SELECT regime, COUNT(*) as cnt, AVG(CAST(hurst AS REAL)) as avg_hurst, AVG(CAST(adx AS REAL)) as avg_adx 
        FROM market_regime 
        GROUP BY regime
    """).fetchall()
    print("Market regime distribution in live data:")
    for rs in reg_summary:
        print(rs)
except Exception as e:
    print(f"Error querying market_regime: {e}")

# Check trades by regime
try:
    reg_trades = conn_data.execute("""
        SELECT regime, COUNT(*) as cnt, 
               SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END)*100.0/COUNT(*) as wr,
               SUM(CAST(pnl AS REAL)) as total_pnl,
               AVG(CAST(pnl AS REAL)) as avg_pnl
        FROM trades 
        WHERE regime IS NOT NULL AND regime != ''
        GROUP BY regime
    """).fetchall()
    print("\nLive trade performance grouped by Market Regime:")
    for rt in reg_trades:
        print(rt)
except Exception as e:
    print(f"Error querying trades by regime: {e}")
