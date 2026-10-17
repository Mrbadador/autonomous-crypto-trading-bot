"""
scripts/reconcile_d4_corrupt_exits.py
Part of FIX P2-9 (D-4).

Reconciles the 9 corrupt exit prices in swing_trades (IDs 149, 152, 156, 163, 166, 168, 170, 172, 192)
where exit prices were corrupted by cross-symbol fill aggregation or oracle price leakage.

Defaults to --dry-run. Requires --apply to execute changes.
When --apply is specified:
  1. Creates backup table 'swing_trades_backup_d4'.
  2. For each corrupt trade, computes plausible implied exit price from (pnl + fee) / size if mathematically sound,
     or marks status='UNRECONCILED' with NULL exit_price / pnl when ground-truth fill price is unrecoverable.
  3. Validates all updated rows using validate_exit_price_plausibility.
"""

import argparse
import os
import re
import sqlite3
import sys

from reconciliation_engine import validate_exit_price_plausibility

TARGET_CORRUPT_IDS = (149, 152, 156, 163, 166, 168, 170, 172, 192)


def reconcile_corrupt_exits(db_path: str, apply_changes: bool = False, target_ids: tuple = TARGET_CORRUPT_IDS) -> int:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found: {db_path}")
        return 1

    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout = 10000;")

        has_table = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='swing_trades'"
        ).fetchone()[0]
        if not has_table:
            print("[ERROR] Table 'swing_trades' does not exist.")
            return 1

        id_placeholders = ",".join("?" for _ in target_ids)
        rows = conn.execute(
            f"SELECT * FROM swing_trades WHERE id IN ({id_placeholders}) ORDER BY id",
            target_ids,
        ).fetchall()

        if not rows:
            print(f"No corrupt rows found with IDs: {target_ids}.")
            return 0

        print(f"Found {len(rows)} target rows to evaluate for reconciliation.\n")

        planned_updates = []
        for r in rows:
            r_id = r["id"]
            sym = r["symbol"]
            side = str(r["side"] or "long").upper()
            entry_p = float(r["entry_price"] or 0.0)
            exit_p = float(r["exit_price"] or 0.0)
            size = float(r["size"] or 0.0)
            pnl = float(r["pnl"]) if r["pnl"] is not None else None
            fee = float(r["fee_paid_usd"] or 0.0)
            notes = str(r["notes"] or "")

            # If fee_paid_usd is 0, attempt parsing fee from notes
            if fee == 0.0 and notes:
                m = re.search(r"Fee[s]?: [-$]?([0-9.]+)", notes)
                if m:
                    fee = float(m.group(1))

            # Attempt implied exit price reconstruction: PnL = (exit - entry)*size - fee
            implied_exit = None
            if size > 0 and entry_p > 0 and pnl is not None:
                if "LONG" in side:
                    implied_exit = entry_p + (pnl + fee) / size
                else:
                    implied_exit = entry_p - (pnl + fee) / size

            # Check if implied_exit is plausible
            is_plausible = (
                implied_exit is not None
                and validate_exit_price_plausibility(sym, side, entry_p, implied_exit, size, pnl)
            )

            if is_plausible:
                # Recalculate accurate ROE using 5x default leverage
                roe = ((implied_exit - entry_p) / entry_p * 500.0) if "LONG" in side else ((entry_p - implied_exit) / entry_p * 500.0)
                reconciled_notes = f"[RECONCILED-D4] Exit price restored from PnL fills | {notes}"
                planned_updates.append({
                    "id": r_id,
                    "symbol": sym,
                    "action": "RESTORE_IMPLIED_PRICE",
                    "old_exit": exit_p,
                    "new_exit": implied_exit,
                    "old_roe": r["roe_pct"],
                    "new_roe": roe,
                    "old_pnl": pnl,
                    "new_pnl": pnl,
                    "new_status": "CLOSED",
                    "notes": reconciled_notes,
                })
            else:
                reconciled_notes = f"[RECONCILED-D4-UNRECONCILED] Implausible cross-fill price purged | {notes}"
                planned_updates.append({
                    "id": r_id,
                    "symbol": sym,
                    "action": "SET_UNRECONCILED",
                    "old_exit": exit_p,
                    "new_exit": None,
                    "old_roe": r["roe_pct"],
                    "new_roe": None,
                    "old_pnl": pnl,
                    "new_pnl": None,
                    "new_status": "UNRECONCILED",
                    "notes": reconciled_notes,
                })

        # Display plan
        print("Reconciliation Action Plan:")
        for u in planned_updates:
            print(
                f"  ID {u['id']:3d} ({u['symbol']}): action={u['action']}, "
                f"exit: {u['old_exit']} -> {u['new_exit']}, status -> {u['new_status']}"
            )

        if not apply_changes:
            print("\n[DRY RUN] No changes applied. Run with --apply to execute reconciliation.")
            return 0

        # Apply mode: Backup then update
        print("\n[APPLY] Creating backup table 'swing_trades_backup_d4'...")
        conn.execute("CREATE TABLE IF NOT EXISTS swing_trades_backup_d4 AS SELECT * FROM swing_trades")
        b_cnt = conn.execute("SELECT count(*) FROM swing_trades_backup_d4").fetchone()[0]
        print(f"[APPLY] Backup table verified with {b_cnt} rows.")

        print(f"[APPLY] Updating {len(planned_updates)} corrupt exit rows...")
        for u in planned_updates:
            conn.execute("""
                UPDATE swing_trades
                SET exit_price = ?, roe_pct = ?, pnl = ?, status = ?, notes = ?
                WHERE id = ?
            """, (u["new_exit"], u["new_roe"], u["new_pnl"], u["new_status"], u["notes"], u["id"]))

        conn.commit()
        print(f"[APPLY] Successfully updated {len(planned_updates)} rows.")

        # Post-verification
        corrupt_rem = conn.execute("""
            SELECT count(*) FROM swing_trades
            WHERE id IN ({})
              AND exit_price IS NOT NULL
              AND (exit_price / entry_price < 0.70 OR exit_price / entry_price > 1.40)
        """.format(id_placeholders), target_ids).fetchone()[0]

        print(f"[VERIFY] Corrupt exit prices remaining among target IDs: {corrupt_rem}")
        if corrupt_rem == 0:
            print("[SUCCESS] All corrupt exit prices resolved cleanly!")
            return 0
        else:
            print(f"[WARNING] {corrupt_rem} rows still have corrupt prices!")
            return 1
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconcile corrupt exit prices in swing_trades.")
    parser.add_argument("--db", default="trading_data.db", help="Path to SQLite database (default: trading_data.db)")
    parser.add_argument("--apply", action="store_true", default=False, help="Apply changes to database (default: dry-run)")
    args = parser.parse_args()

    sys.exit(reconcile_corrupt_exits(args.db, apply_changes=args.apply))
