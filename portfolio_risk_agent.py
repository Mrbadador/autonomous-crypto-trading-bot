import sqlite3
import time
import logging
import json
import traceback
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] RISK_AGENT: %(message)s")
log = logging.getLogger("risk_agent")

MAX_OPEN_POSITIONS = getattr(config, "GLOBAL_MAX_POSITIONS", 5)
MAX_PORTFOLIO_EXPOSURE = getattr(config, "INITIAL_BALANCE", 1000.0) * 1.5

def get_positions(cursor, table_name):
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [c[1] for c in cursor.fetchall()]
        if not cols:
            return []
        has_lev = "leverage" in cols
        if has_lev:
            cursor.execute(f"SELECT symbol, entry_price, qty, leverage FROM {table_name}")
            return [(r[0], float(r[1] or 0), float(r[2] or 0), float(r[3] or 5.0)) for r in cursor.fetchall()]
        else:
            cursor.execute(f"SELECT symbol, entry_price, qty FROM {table_name}")
            return [(r[0], float(r[1] or 0), float(r[2] or 0), float(getattr(config, "LEVERAGE", 5.0))) for r in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise

def get_current_price(cursor, symbol, entry_price):
    try:
        cursor.execute("SELECT price FROM cycles WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1", (symbol,))
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except sqlite3.OperationalError:
        pass
    return entry_price

def run_risk_agent():
    log.info(f"Portfolio Risk Agent starting... (Max Positions: {MAX_OPEN_POSITIONS} | Max Exposure: ${MAX_PORTFOLIO_EXPOSURE:.2f})")
    while True:
        try:
            with sqlite3.connect("trading_data.db", timeout=10) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                cursor = conn.cursor()
                
                sniper_pos = get_positions(cursor, "open_positions")
                berserker_pos = get_positions(cursor, "berserker_open_positions")
                mr_pos = get_positions(cursor, "mean_reversion_open_positions")
                try:
                    cursor.execute("SELECT symbol, entry_price, size AS qty, leverage FROM swing_open_positions WHERE status = 'ACTIVE'")
                    swing_pos = [(r[0], float(r[1] or 0), float(r[2] or 0), float(r[3] or 3.0)) for r in cursor.fetchall()]
                except Exception:
                    swing_pos = []
                
                all_pos = sniper_pos + berserker_pos + mr_pos + swing_pos
                
                total_exposure = 0.0
                total_margin = 0.0
                significant_coins = set()
                MIN_SIGNIFICANT_MARGIN = 0.05  # Positions with margin < $0.05 (e.g. runners/dust) don't count toward max slots

                for sym, ep, qty, lev in all_pos:
                    curr_p = get_current_price(cursor, sym, ep)
                    notional = curr_p * qty
                    total_exposure += notional
                    pos_lev = max(1.0, float(lev or 3.0))
                    margin = notional / pos_lev
                    total_margin += margin
                    if margin >= MIN_SIGNIFICANT_MARGIN:
                        sym_clean = sym.replace("/", "").replace("-", "").upper()
                        significant_coins.add(sym_clean)
                    
                total_count = len(significant_coins)

                # Read dynamic max positions limit from system_state
                cursor.execute("SELECT value FROM system_state WHERE key = 'GLOBAL_MAX_POSITIONS'")
                row_max = cursor.fetchone()
                max_open = int(row_max[0]) if row_max and row_max[0].isdigit() else MAX_OPEN_POSITIONS

                # Read dynamic max exposure based on live balance (70% portfolio margin ceiling)
                cursor.execute("SELECT value FROM system_state WHERE key = 'balance'")
                row_bal = cursor.fetchone()
                live_bal = float(row_bal[0]) if row_bal else (MAX_PORTFOLIO_EXPOSURE / getattr(config, "LEVERAGE", 5.0))
                max_margin_pct = getattr(config, "GLOBAL_MARGIN_LIMIT", 70.0) / 100.0
                max_margin_allowed = live_bal * max_margin_pct

                is_halted = total_count >= max_open or total_margin >= max_margin_allowed
                
                # Write to system_state
                halt_val = "1" if is_halted else "0"
                cursor.execute("""
                    INSERT INTO system_state (key, value, updated_at) 
                    VALUES ('RISK_HALT', ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """, (halt_val,))
                
                # Also save risk metrics for HUD
                metrics = {
                    "total_exposure": total_exposure,
                    "total_margin": total_margin,
                    "max_margin_allowed": max_margin_allowed,
                    "total_count": total_count,
                    "max_open": max_open,
                    "halted": is_halted
                }
                cursor.execute("""
                    INSERT INTO system_state (key, value, updated_at) 
                    VALUES ('risk_metrics', ?, datetime('now'))
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """, (json.dumps(metrics),))
                
                conn.commit()
                
        except Exception as e:
            log.error(f"Error in Risk Agent loop: {e}")
            traceback.print_exc()
            
        time.sleep(5)

if __name__ == "__main__":
    run_risk_agent()
