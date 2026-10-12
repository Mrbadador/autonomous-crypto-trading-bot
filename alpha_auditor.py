import sqlite3
import logging
import os
import sys
import pandas as pd
from datetime import datetime

# [PATCH V32.4] Ghost trade table DDL for Shadow-Rehabilitation
GHOST_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS ghost_trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    entry_price REAL    NOT NULL,
    exit_price  REAL,
    pnl         REAL,
    outcome     TEXT,
    is_ghost    INTEGER DEFAULT 1,
    signal_type TEXT    DEFAULT 'Sniper-Strategy',
    paper_tp    REAL,
    paper_sl    REAL
);"""

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] alpha_auditor: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/alpha_auditor.log")
    ]
)
log = logging.getLogger("alpha_auditor")

DB_PATH = "trading_data.db"

def run_audit():
    log.info("=== Starting V26.1 Forensic Alpha Audit ===")
    
    if not os.path.exists(DB_PATH):
        log.error(f"Database not found at {DB_PATH}. Aborting.")
        return

    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        
        # [PATCH V32.4] Ensure ghost_trades schema is migrated with paper_tp and paper_sl
        conn.execute(GHOST_TABLE_DDL)
        for col in ["paper_tp", "paper_sl"]:
            try:
                conn.execute(f"ALTER TABLE ghost_trades ADD COLUMN {col} REAL")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        # 1. Load and Combine Active Trade Datasets (Exclude old pre-patch legacy_trades)
        log.info("Querying trade tables for active performance metrics...")
        
        query_sniper = "SELECT timestamp, pnl, 'Sniper-Strategy' as signal_type FROM trades WHERE timestamp >= datetime('now', '-14 days') AND pnl IS NOT NULL AND type IN ('sl_hit', 'tp_hit', 'closed')"
        query_berserker = "SELECT timestamp, pnl, notes, 'Berserker-Strategy' as signal_type FROM berserker_trades WHERE timestamp >= datetime('now', '-14 days') AND pnl IS NOT NULL AND type IN ('sl_hit', 'tp_hit', 'closed')"
        
        df_sniper = pd.read_sql_query(query_sniper, conn)
        df_berserker = pd.read_sql_query(query_berserker, conn)
        
        # V26.2 Iceberg Post-Processing Audit
        if not df_berserker.empty:
            # Tag rows containing the unique metadata flag as a distinct forensic segment
            mask = df_berserker['notes'].fillna('').astype(str).str.contains(r'\[ICEBERG\]', regex=True)
            count_found = int(mask.sum())
            if count_found > 0:
                log.info(f"Located {count_found} 'Iceberg-Triggered' trades for partitioned tracking.")
                df_berserker.loc[mask, 'signal_type'] = 'Iceberg-Triggered'
        
        # Ensure consistent types
        all_dfs = []
        for df in [df_sniper, df_berserker]:
            if not df.empty:
                df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                all_dfs.append(df.dropna(subset=['pnl']))
                
        if not all_dfs:
            log.info("No active session trades yet. Clearing kill-switches for all strategies.")
            conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('DISABLED_SIGNAL_Sniper-Strategy', '0')")
            conn.execute("INSERT OR REPLACE INTO system_state (key, value) VALUES ('DISABLED_SIGNAL_Berserker-Strategy', '0')")
            conn.commit()
            conn.close()
            return
            
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df = master_df.sort_values(by='timestamp').reset_index(drop=True)
        
        log.info(f"Aggregated {len(master_df)} total processed records for forensic analysis.")
        
        # 2. Group and Audit each Signal Type
        signal_groups = master_df.groupby('signal_type')
        
        for signal_type, group in signal_groups:
            log.info(f"Processing Auditor logic for Signal Group: [{signal_type}]")
            
            # Sort subgroup and get top 50 most recent
            sorted_group = group.sort_values(by='timestamp', ascending=True)
            sample_window = sorted_group.tail(50)
            sample_count = len(sample_window)
            
            if sample_count == 0:
                log.warning(f"Skipping {signal_type}, sample count is 0.")
                continue
                
            # Calculate Stats
            wins = sample_window[sample_window['pnl'] > 0]
            losses = sample_window[sample_window['pnl'] < 0]
            
            win_count = len(wins)
            win_rate = (win_count / sample_count) * 100.0
            
            gross_profit = wins['pnl'].sum()
            gross_loss = abs(losses['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            log.info(f"[{signal_type}] Last {sample_count} Trades -> WinRate: {win_rate:.2f}%, Profit Factor: {profit_factor:.2f}")
            
            # 3. Apply Kill-Switch Logic
            # Logic: If win rate drops below 45% (min 10 sample threshold to prevent noise early on, optional but safe)
            # User asked for specific threshold over the last 50 samples.
            
            status_key = f"DISABLED_SIGNAL_{signal_type}"
            
            if sample_count >= 10 and win_rate < 45.0:
                log.warning(f"CRITICAL: [{signal_type}] performance breach! Win Rate {win_rate:.1f}% is BELOW threshold of 45%. Writing DISABLED_SIGNAL flag to DB.")
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                    (status_key, "1")
                )
            else:
                log.info(f"[{signal_type}] passed compliance. Ensuring kill-switch is cleared.")
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                    (status_key, "0")
                )
        
        # --- [PATCH V32.2-P3] SHADOW-REHABILITATION ---
        # Ensure ghost_trades table exists.
        conn.execute(GHOST_TABLE_DDL)

        sniper_disabled = conn.execute(
            "SELECT value FROM system_state WHERE key = 'DISABLED_SIGNAL_Sniper-Strategy'"
        ).fetchone()

        if sniper_disabled and sniper_disabled[0] == '1':
            ghost_df = pd.read_sql_query(
                # [V32.3] Only count fully CLOSED ghost positions (pnl AND outcome both set)
                "SELECT pnl, outcome FROM ghost_trades "
                "WHERE signal_type='Sniper-Strategy' "
                "AND pnl IS NOT NULL AND outcome IS NOT NULL "
                "ORDER BY id DESC LIMIT 20",
                conn
            )
            if len(ghost_df) >= 10:
                ghost_wins = len(ghost_df[ghost_df['pnl'] > 0])
                ghost_wr = (ghost_wins / len(ghost_df)) * 100.0
                tp_count = int((ghost_df['outcome'] == 'tp_hit').sum())
                sl_count = int((ghost_df['outcome'] == 'sl_hit').sum())
                log.info(
                    f"[GHOST-REHAB] Sniper ghost WR over last {len(ghost_df)} closed signals: "
                    f"{ghost_wr:.1f}% | Breakdown: {tp_count} TP hits, {sl_count} SL hits"
                )
                if ghost_wr >= 45.0:
                    log.warning(
                        "[GHOST-REHAB] REHABILITATION THRESHOLD MET. "
                        "Clearing Sniper kill-switch. Strategy re-enabled."
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO system_state (key, value) VALUES (?, ?)",
                        ('DISABLED_SIGNAL_Sniper-Strategy', '0')
                    )
                else:
                    log.info(
                        f"[GHOST-REHAB] Ghost WR {ghost_wr:.1f}% still below 45%. Kill-switch maintained."
                    )
            else:
                # Count open (unresolved) ghost entries separately for diagnostics
                open_count = conn.execute(
                    "SELECT COUNT(*) FROM ghost_trades "
                    "WHERE signal_type='Sniper-Strategy' AND outcome IS NULL"
                ).fetchone()[0]
                log.info(
                    f"[GHOST-REHAB] Only {len(ghost_df)} closed ghost signals "
                    f"(need 10). {open_count} still open/pending resolution."
                )
        # --- END PATCH V32.2-P3 / V32.3 ---

        conn.commit()
        conn.close()
        log.info("Alpha Audit writeback finalized. Database synchronized.")
        
    except Exception as e:
        log.exception(f"Alpha Auditor encountered fatal logic execution error: {e}")

if __name__ == "__main__":
    run_audit()
