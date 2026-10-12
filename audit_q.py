import sqlite3

print("=== QUERY 1: training/fast_gate/marl/policy flags ===")
conn = sqlite3.connect("trading_data.db")
rows = conn.execute("SELECT key, value, updated_at FROM system_state WHERE key LIKE ? OR key LIKE ? OR key LIKE ? OR key LIKE ?", ("%training%", "%fast_gate%", "%marl%", "%policy%")).fetchall()
for r in rows:
    print(r)
print("---")
rows2 = conn.execute("SELECT key, SUBSTR(value,1,200), updated_at FROM system_state ORDER BY updated_at DESC LIMIT 20").fetchall()
for r in rows2:
    print(r)

print("\n=== QUERY 2: HWM drawdown state ===")
keys = ("hwm_balance","peak_balance","current_drawdown","max_drawdown","session_drawdown","marl_live_risk_cap","marl_live_gamma")
placeholders = ",".join("?" * len(keys))
r2 = conn.execute(f"SELECT key, value FROM system_state WHERE key IN ({placeholders})", keys).fetchall()
for r in r2:
    print(r)

print("\n=== QUERY 3: Win rate by symbol ===")
conn.row_factory = sqlite3.Row
rows3 = conn.execute("SELECT symbol, SUM(CASE WHEN CAST(pnl AS REAL)>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) as wr, AVG(CAST(pnl AS REAL)) as avg_pnl, COUNT(*) as n FROM trades WHERE status='closed' GROUP BY symbol ORDER BY n DESC").fetchall()
for r in rows3:
    print(dict(r))

print("\n=== QUERY 4: Swing trades ===")
try:
    rows4 = conn.execute("SELECT symbol, side, pnl, exit_reason, entry_time, exit_time FROM swing_trades ORDER BY entry_time DESC LIMIT 20").fetchall()
    for r in rows4:
        print(dict(r))
except Exception as e:
    print(f"swing_trades query error: {e}")

conn.close()

print("\n=== QUERY 5: Learned rules ===")
try:
    conn2 = sqlite3.connect("trading_ledger.db")
    r5 = conn2.execute("SELECT COUNT(*) FROM learned_rules").fetchone()
    print(f"Total learned rules: {r5[0]}")
    rows5 = conn2.execute("SELECT rule_id, rule_text, confidence, source, created_at FROM learned_rules ORDER BY confidence DESC LIMIT 10").fetchall()
    for r in rows5:
        print(r)
    conn2.close()
except Exception as e:
    print(f"learned_rules query error: {e}")
