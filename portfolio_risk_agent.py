import os
import sqlite3
import time
import logging
import json
import traceback
from datetime import datetime, UTC
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] RISK_AGENT: %(message)s")
log = logging.getLogger("risk_agent")

MAX_OPEN_POSITIONS = int(getattr(config, "GLOBAL_MAX_POSITIONS", 8))
FALLBACK_BALANCE = float(getattr(config, "INITIAL_BALANCE", 250.0))
MAX_EXPOSURE_MULT = float(getattr(config, "MAX_PORTFOLIO_EXPOSURE_MULT", 3.5))
MICROSTRUCTURE_STATE_FILE = "microstructure_state.json"


def get_positions(cursor, table_name: str) -> list[tuple[str, float, float, float]]:
    """
    Loads active, non-closing positions with positive quantity from any fleet position table.
    Supports both `qty` (Sniper/Berserker/MARL/MR) and `size` (Swing) schemas.
    """
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [c[1] for c in cursor.fetchall()]
        if not cols:
            return []

        qty_col = "size" if ("size" in cols and "qty" not in cols) else ("qty" if "qty" in cols else None)
        if not qty_col:
            return []

        has_lev = "leverage" in cols
        where_clauses = [f"CAST(COALESCE({qty_col}, 0) AS REAL) > 0"]
        if "status" in cols:
            where_clauses.append("UPPER(COALESCE(status, 'ACTIVE')) = 'ACTIVE'")
        if "is_closing" in cols:
            where_clauses.append("COALESCE(is_closing, 0) = 0")

        where_sql = " AND ".join(where_clauses)
        default_lev = 3.0 if "swing" in table_name else float(getattr(config, "LEVERAGE", 5.0))

        if has_lev:
            cursor.execute(f"SELECT symbol, entry_price, {qty_col}, leverage FROM {table_name} WHERE {where_sql}")
            return [
                (str(r[0]), float(r[1] or 0.0), float(r[2] or 0.0), float(r[3] or default_lev))
                for r in cursor.fetchall()
                if r[0]
            ]
        else:
            cursor.execute(f"SELECT symbol, entry_price, {qty_col} FROM {table_name} WHERE {where_sql}")
            return [
                (str(r[0]), float(r[1] or 0.0), float(r[2] or 0.0), default_lev)
                for r in cursor.fetchall()
                if r[0]
            ]
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return []
        raise


def load_live_micro_prices() -> dict[str, float]:
    """Reads live Bitget mid/micro prices from `microstructure_state.json` if fresh (<= 300s)."""
    prices: dict[str, float] = {}
    if not os.path.exists(MICROSTRUCTURE_STATE_FILE):
        return prices
    try:
        with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = float(data.get("timestamp", 0.0) or 0.0)
        if ts > 0 and abs(time.time() - ts) > 300.0:
            return prices
        symbols_map = data.get("symbols", {})
        if isinstance(symbols_map, dict):
            for sym_key, info in symbols_map.items():
                if not isinstance(info, dict):
                    continue
                mp = float(info.get("micro_price", 0.0) or 0.0)
                if mp <= 0:
                    bid = float(info.get("best_bid", 0.0) or 0.0)
                    ask = float(info.get("best_ask", 0.0) or 0.0)
                    if bid > 0 and ask > 0:
                        mp = (bid + ask) / 2.0
                if mp > 0:
                    prices[sym_key.replace("/", "").replace("-", "").upper()] = mp
    except Exception:
        pass
    return prices


def get_current_price(cursor, symbol: str, entry_price: float, live_prices: dict[str, float]) -> float:
    """Marks position to market using live OFI daemon feed first, then `cycles`, then `entry_price`."""
    sym_clean = symbol.replace("/", "").replace("-", "").upper()
    if sym_clean in live_prices and live_prices[sym_clean] > 0:
        return live_prices[sym_clean]
    try:
        cursor.execute(
            "SELECT price FROM cycles WHERE symbol = ? OR REPLACE(symbol, '/', '') = ? ORDER BY timestamp DESC LIMIT 1",
            (symbol, sym_clean),
        )
        row = cursor.fetchone()
        if row and row[0] is not None and float(row[0]) > 0:
            return float(row[0])
    except sqlite3.OperationalError:
        pass
    return entry_price


def run_risk_agent():
    max_margin_pct_cfg = float(getattr(config, "GLOBAL_MARGIN_LIMIT", 70.0))
    log.info(
        f"Portfolio Risk Agent V43.0 starting... "
        f"(Max Slots: {MAX_OPEN_POSITIONS} | Margin Cap: {max_margin_pct_cfg:.0f}% | Max Exposure Mult: {MAX_EXPOSURE_MULT:.1f}x)"
    )

    prev_halted: bool | None = None
    prev_reasons: list[str] = []

    fleet_tables = [
        "open_positions",
        "berserker_open_positions",
        "mean_reversion_open_positions",
        "marl_live_open_positions",
        "swing_open_positions",
    ]

    while True:
        try:
            live_prices = load_live_micro_prices()
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                cursor = conn.cursor()

                all_pos: list[tuple[str, float, float, float]] = []
                for tbl in fleet_tables:
                    all_pos.extend(get_positions(cursor, tbl))

                total_exposure = 0.0
                total_margin = 0.0
                significant_coins = set()
                active_positions_count = 0
                MIN_SIGNIFICANT_MARGIN = 0.05  # Ignore dust (< $0.05 margin)

                for sym, ep, qty, lev in all_pos:
                    curr_p = get_current_price(cursor, sym, ep, live_prices)
                    notional = curr_p * qty
                    total_exposure += notional
                    pos_lev = max(1.0, float(lev or 3.0))
                    margin = notional / pos_lev
                    total_margin += margin
                    if margin >= MIN_SIGNIFICANT_MARGIN:
                        active_positions_count += 1
                        sym_clean = sym.replace("/", "").replace("-", "").upper()
                        significant_coins.add(sym_clean)

                unique_coins_count = len(significant_coins)
                # Count each active position slot (so 2 bots on BTC = 2 slots, not 1)
                total_count = max(active_positions_count, unique_coins_count)

                # Read dynamic max positions limit from system_state
                cursor.execute("SELECT value FROM system_state WHERE key = 'GLOBAL_MAX_POSITIONS'")
                row_max = cursor.fetchone()
                try:
                    max_open = int(row_max[0]) if (row_max and str(row_max[0]).strip().isdigit()) else MAX_OPEN_POSITIONS
                except Exception:
                    max_open = MAX_OPEN_POSITIONS

                # Read live balance from system_state (with clean fallback to config.INITIAL_BALANCE)
                cursor.execute("SELECT value FROM system_state WHERE key = 'balance'")
                row_bal = cursor.fetchone()
                try:
                    live_bal = float(row_bal[0]) if (row_bal and row_bal[0] is not None and float(row_bal[0]) > 0) else FALLBACK_BALANCE
                except Exception:
                    live_bal = FALLBACK_BALANCE

                max_margin_pct = float(getattr(config, "GLOBAL_MARGIN_LIMIT", 70.0)) / 100.0
                max_margin_allowed = live_bal * max_margin_pct
                max_exposure_allowed = live_bal * MAX_EXPOSURE_MULT

                # Read sticky manual operator pause (from Dashboard / Mobile HUD)
                cursor.execute("SELECT value FROM system_state WHERE key = 'OPERATOR_HALT'")
                row_op = cursor.fetchone()
                operator_halt = bool(row_op and str(row_op[0]).strip().upper() in ("1", "TRUE"))

                # Hysteresis State Machine for Margin & Exposure Cap:
                # Engage halt at 70% margin limit; clear only below 62% margin limit to eliminate flapping.
                margin_clear_pct = max(0.20, max_margin_pct - 0.08)
                max_margin_clear = live_bal * margin_clear_pct

                exposure_clear_mult = max(1.0, MAX_EXPOSURE_MULT - 0.5)
                max_exposure_clear = live_bal * exposure_clear_mult

                # Determine if auto halt was previously active
                cursor.execute("SELECT value FROM system_state WHERE key = 'AUTO_RISK_HALTED'")
                row_auto_prev = cursor.fetchone()
                auto_halt_prev = bool(row_auto_prev and str(row_auto_prev[0]).strip() == "1")

                halt_reasons: list[str] = []
                if operator_halt:
                    halt_reasons.append("OPERATOR_PAUSE")

                auto_halted = auto_halt_prev
                if not auto_halt_prev:
                    # Condition to engage halt
                    if total_count >= max_open:
                        auto_halted = True
                        halt_reasons.append(f"SLOT_CAP({total_count}/{max_open})")
                    if total_margin >= max_margin_allowed:
                        auto_halted = True
                        halt_reasons.append(f"MARGIN_CAP(${total_margin:.2f}>=${max_margin_allowed:.2f})")
                    if total_exposure >= max_exposure_allowed:
                        auto_halted = True
                        halt_reasons.append(f"EXPOSURE_CAP(${total_exposure:.2f}>=${max_exposure_allowed:.2f})")
                else:
                    # Currently halted: check if cleared below hysteresis buffer
                    if total_count < max_open and total_margin < max_margin_clear and total_exposure < max_exposure_clear:
                        auto_halted = False
                    else:
                        if total_count >= max_open:
                            halt_reasons.append(f"SLOT_CAP({total_count}/{max_open})")
                        if total_margin >= max_margin_clear:
                            halt_reasons.append(f"MARGIN_HYSTERESIS(${total_margin:.2f}>=${max_margin_clear:.2f})")
                        if total_exposure >= max_exposure_clear:
                            halt_reasons.append(f"EXPOSURE_HYSTERESIS(${total_exposure:.2f}>=${max_exposure_clear:.2f})")

                is_halted = operator_halt or auto_halted

                # Log state transitions clearly
                if prev_halted is None or is_halted != prev_halted or halt_reasons != prev_reasons:
                    if is_halted:
                        log.warning(
                            f"[HALT-ENGAGED] 🛑 RISK_HALT=1 | Reasons: {', '.join(halt_reasons)} | "
                            f"Positions: {active_positions_count} ({unique_coins_count} coins) / {max_open} | "
                            f"Margin: ${total_margin:.2f} / ${max_margin_allowed:.2f} | "
                            f"Exposure: ${total_exposure:.2f} / ${max_exposure_allowed:.2f}"
                        )
                    else:
                        log.info(
                            f"[HALT-CLEARED] 🟢 RISK_HALT=0 (Fleet Entry Gate Open) | "
                            f"Positions: {active_positions_count} ({unique_coins_count} coins) / {max_open} | "
                            f"Margin: ${total_margin:.2f} / ${max_margin_allowed:.2f} | "
                            f"Exposure: ${total_exposure:.2f} / ${max_exposure_allowed:.2f}"
                        )
                    prev_halted = is_halted
                    prev_reasons = list(halt_reasons)

                now_iso = datetime.now(UTC).isoformat()
                halt_val = "1" if is_halted else "0"
                auto_halt_val = "1" if auto_halted else "0"
                cursor.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES ('RISK_HALT', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (halt_val, now_iso),
                )
                cursor.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES ('AUTO_RISK_HALTED', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (auto_halt_val, now_iso),
                )

                metrics = {
                    "total_exposure": round(total_exposure, 2),
                    "max_exposure_allowed": round(max_exposure_allowed, 2),
                    "total_margin": round(total_margin, 2),
                    "max_margin_allowed": round(max_margin_allowed, 2),
                    "total_count": total_count,
                    "active_positions": active_positions_count,
                    "unique_coins": unique_coins_count,
                    "max_open": max_open,
                    "operator_halt": operator_halt,
                    "auto_halted": auto_halted,
                    "halt_reasons": halt_reasons,
                    "halted": is_halted,
                    "updated_at": now_iso,
                }
                cursor.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES ('risk_metrics', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (json.dumps(metrics), now_iso),
                )

                conn.commit()

        except Exception as e:
            log.error(f"Error in Risk Agent loop: {e}")
            traceback.print_exc()

        time.sleep(5)


if __name__ == "__main__":
    run_risk_agent()
