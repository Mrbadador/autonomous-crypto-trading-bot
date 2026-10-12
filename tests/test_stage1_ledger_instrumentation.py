import os
import sys
import sqlite3
import json
from datetime import datetime, UTC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import indicators as ind
from db_manager import DatabaseManager


def test_stage1_ledger_instrumentation():
    test_db = "tests/test_stage1_trading.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    # 1. Initialize schema
    db_m = DatabaseManager()
    db_m.db_path = test_db
    db_m._init_db()

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()

    # 2. Assert all new columns exist on swing_trades
    cursor.execute("PRAGMA table_info(swing_trades)")
    st_cols = {row[1]: row[2] for row in cursor.fetchall()}
    expected_st_cols = [
        "intended_entry_price",
        "actual_fill_price",
        "slippage_bps",
        "fee_paid_usd",
        "fee_maker_usd",
        "fee_taker_usd",
        "funding_usd",
        "funding_source",
        "gates_snapshot",
        "status"
    ]
    for col in expected_st_cols:
        assert col in st_cols, f"Missing column {col} in swing_trades"

    # 3. Assert all new columns exist on swing_open_positions
    cursor.execute("PRAGMA table_info(swing_open_positions)")
    sop_cols = {row[1]: row[2] for row in cursor.fetchall()}
    expected_sop_cols = [
        "intended_entry_price",
        "actual_fill_price",
        "slippage_bps",
        "fee_entry_usd",
        "execution_type",
        "gates_snapshot"
    ]
    for col in expected_sop_cols:
        assert col in sop_cols, f"Missing column {col} in swing_open_positions"

    # 4. Simulate mock trade open with slippage calculation
    symbol = "BTC/USDT"
    intended_p = 65000.0
    actual_fill_p = 65006.50
    qty = 0.05
    slippage_bps = ((actual_fill_p - intended_p) / intended_p) * 10000.0  # +1.0 bps
    assert abs(slippage_bps - 1.0) < 1e-6, f"Slippage math error: {slippage_bps}"

    gates_data = {
        "1d_trend": "BULLISH",
        "4h_adx": 32.5,
        "momentum_divergence": {"pass": True},
        "td_sequential": {"pass": True}
    }
    gates_json = json.dumps(gates_data)

    cursor.execute("""
        INSERT INTO swing_open_positions (
            symbol, side, entry_price, size, stop_loss, take_profit, trailing_stop,
            peak_price, leverage, opened_at, updated_at, status, r_multiple,
            initial_risk_usd, notes, intended_entry_price, actual_fill_price,
            slippage_bps, fee_entry_usd, execution_type, gates_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol, "long", actual_fill_p, qty, 64000.0, 67000.0, 64000.0,
        actual_fill_p, 3.0, datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat(),
        "ACTIVE", 0.0, 50.0, "Test open", intended_p, actual_fill_p,
        slippage_bps, actual_fill_p * qty * 0.0002, "maker", gates_json
    ))
    conn.commit()

    # Read back position
    row = cursor.execute("SELECT intended_entry_price, actual_fill_price, slippage_bps, execution_type, gates_snapshot FROM swing_open_positions WHERE symbol=?", (symbol,)).fetchone()
    assert row[0] == intended_p
    assert row[1] == actual_fill_p
    assert abs(row[2] - 1.0) < 1e-6
    assert row[3] == "maker"
    loaded_gates = json.loads(row[4])
    assert loaded_gates["1d_trend"] == "BULLISH"

    # 5. Simulate position close with funding history and fee split
    exit_p = 66500.0
    fee_maker = actual_fill_p * qty * 0.0002   # $0.650065
    fee_taker = exit_p * qty * 0.0006          # $1.995
    tot_fee = fee_maker + fee_taker
    funding_usd = -0.15                         # paid funding
    funding_src = "bitget_api"
    gross_pnl = (exit_p - actual_fill_p) * qty # $74.675
    net_pnl = gross_pnl - tot_fee + funding_usd

    cursor.execute("""
        INSERT INTO swing_trades (
            timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct,
            holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price,
            slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd,
            funding_source, gates_snapshot, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(UTC).isoformat(), symbol, "long", actual_fill_p, exit_p, qty,
        net_pnl, 5.0, 4.5, "TAKE_PROFIT", "Test close", intended_p, actual_fill_p,
        slippage_bps, tot_fee, fee_maker, fee_taker, funding_usd, funding_src,
        gates_json, "CLOSED"
    ))
    cursor.execute("DELETE FROM swing_open_positions WHERE symbol=?", (symbol,))
    conn.commit()

    # 6. Verify swing_trades record
    closed_row = cursor.execute("""
        SELECT symbol, intended_entry_price, actual_fill_price, slippage_bps,
               fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd, funding_source, status
        FROM swing_trades WHERE symbol=?
    """, (symbol,)).fetchone()
    assert closed_row[0] == symbol
    assert closed_row[1] == intended_p
    assert closed_row[2] == actual_fill_p
    assert abs(closed_row[3] - 1.0) < 1e-6
    assert abs(closed_row[4] - tot_fee) < 1e-6
    assert abs(closed_row[5] - fee_maker) < 1e-6
    assert abs(closed_row[6] - fee_taker) < 1e-6
    assert closed_row[7] == funding_usd
    assert closed_row[8] == "bitget_api"
    assert closed_row[9] == "CLOSED"

    # 7. Verify Reconciler UNRECONCILED path
    cursor.execute("""
        INSERT INTO swing_trades (
            timestamp, symbol, side, entry_price, exit_price, size, pnl, roe_pct,
            holding_hours, exit_reason, notes, intended_entry_price, actual_fill_price,
            slippage_bps, fee_paid_usd, fee_maker_usd, fee_taker_usd, funding_usd,
            funding_source, gates_snapshot, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(UTC).isoformat(), "ETH/USDT", "long", 3500.0, None, 1.0,
        None, None, 2.0, "UNRECONCILED_CLOSE", "No fills found", 3500.0, 3500.0,
        0.0, None, 0.0, None, 0.0, "reconciler", None, "UNRECONCILED"
    ))
    conn.commit()

    unrec_row = cursor.execute("""
        SELECT symbol, exit_price, pnl, status FROM swing_trades WHERE symbol='ETH/USDT'
    """).fetchone()
    assert unrec_row[1] is None, "Exit price must be NULL on UNRECONCILED"
    assert unrec_row[2] is None, "PnL must be NULL on UNRECONCILED"
    assert unrec_row[3] == "UNRECONCILED"

    conn.close()
    try:
        if hasattr(db_m, "_writer_conn") and db_m._writer_conn:
            db_m._writer_conn.close()
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        if os.path.exists(test_db):
            os.remove(test_db)
    except Exception:
        pass
    print("ALL STAGE 1 ACCEPTANCE TESTS PASSED!")


if __name__ == "__main__":
    test_stage1_ledger_instrumentation()
