import sqlite3

def check_db():
    db1 = sqlite3.connect('trading_data.db')
    print("trading_data.db tables:")
    for row in db1.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('marl_live_trades', 'berserker_trades', 'system_state')").fetchall():
        print(row)
        
    db2 = sqlite3.connect('trading_ledger.db')
    print("\ntrading_ledger.db tables:")
    for row in db2.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('oracle_state', 'sentiment_logs', 'macro_logs')").fetchall():
        print(row)

check_db()
