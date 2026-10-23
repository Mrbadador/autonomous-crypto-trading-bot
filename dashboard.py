"""
dashboard.py — Sovereign Swarm Institutional Command Center HUD (V36.2 Enterprise)
High-Contrast, Real-Time Trading, Microstructure, Calibration, Risk & Asset Breakdown Dashboard.
Includes Auto-Refresh, Dual-Layer Stop Loss, Liquidation Distance, Margin Gauge, Chronological Curves & Operator Drawer.
"""

import os
import sys
import time
import json
import html  # L-1 FIX: HTML escaping for unsafe_allow_html=True blocks
import sqlite3
import pytz
import requests
import socket
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import config
from conviction_filter import check_database_integrity
from indicators import (
    detect_momentum_divergence,
    compute_volume_profile,
    compute_td_sequential
)

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

try:
    from executive_briefing import build_briefing_sync, save_briefing, run_single_briefing_cycle
    BRIEFING_ENGINE_AVAILABLE = True
except Exception:
    BRIEFING_ENGINE_AVAILABLE = False

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign Swarm — Command HUD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths & SQLite Connections ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADING_DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
TRADING_LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")
MICROSTRUCTURE_STATE_FILE = os.path.join(BASE_DIR, "microstructure_state.json")

def connect_readonly(db_path: str, timeout: float = 30.0) -> sqlite3.Connection:
    """Connects to SQLite in WAL Read-Only mode to eliminate lock contention."""
    db_uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(db_uri, timeout=timeout, check_same_thread=False, uri=True)
    return conn

# Feed failures captured during the current script run; surfaced as a banner next rerun.
FEED_FAILURES: dict = {}


def _feed_key(db_path: str, query: str) -> str:
    parts = query.split()
    table = "?"
    upper_parts = [p.upper() for p in parts]
    if "FROM" in upper_parts:
        nxt = upper_parts.index("FROM") + 1
        if nxt < len(parts):
            table = parts[nxt].strip("(").split(".")[-1]
    return f"{os.path.basename(db_path)} \u2192 {table}"


@st.cache_data(ttl=4, show_spinner=False)
def query_db(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    """Safely executes a read-only query and returns a pandas DataFrame.

    Never raises: an unavailable feed returns an empty frame and is recorded in
    FEED_FAILURES so the UI can report the outage instead of silently showing zero.
    """
    if not os.path.exists(db_path):
        FEED_FAILURES[_feed_key(db_path, query)] = "database file missing"
        return pd.DataFrame()
    try:
        conn = connect_readonly(db_path)
        try:
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
        return df
    except Exception as q_err:
        FEED_FAILURES[_feed_key(db_path, query)] = f"{type(q_err).__name__}: {str(q_err)[:120]}"
        return pd.DataFrame()

@st.cache_data(ttl=4, show_spinner=False)
def query_system_state(key: str, default: str = "") -> str:
    """Fetches a single key value from the system_state table."""
    df = query_db(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = ?", (key,))
    if not df.empty and df.iloc[0]["value"] is not None:
        return str(df.iloc[0]["value"])
    return default


@st.cache_data(ttl=4, show_spinner=False)
def query_system_state_row(key: str) -> tuple:
    """Fetch one system_state row as ``(value, updated_at)``, or ``("", "")`` when it is absent.

    `query_system_state` above returns the value only, so a panel reading through it cannot see the
    row's publish instant at all. That is the whole of finding B-1: the Markov deck printed a row
    `regime_classifier.py:1218-1221` wrote ~35 hours earlier under a heading claiming a live forward
    filter, while every automated consumer refuses that row beyond `config.REGIME_MAX_AGE_SEC`.

    The signature of `query_system_state` is deliberately left alone: it is called across this whole
    file, and one panel's age path is not a branch-wide blast radius. Same read-only connection and
    the same 4s cache as the value-only reader.
    """
    try:
        conn = connect_readonly(TRADING_DATA_DB)
        try:
            row = conn.execute(
                "SELECT value, updated_at FROM system_state WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
    except Exception as r_err:
        FEED_FAILURES[f"read \u2192 {os.path.basename(TRADING_DATA_DB)}:{key}"] = (
            f"{type(r_err).__name__}: {str(r_err)[:120]}")
        return "", ""
    if row is None:
        return "", ""
    value = "" if row[0] is None else str(row[0])
    updated_at = "" if row[1] is None else str(row[1])
    return value, updated_at

_JUNK_TOKENS = ("", "none", "null", "nan", "n/a", "na", "syncing...", "--")


def num(val, default: float = 0.0):
    """Coerce telemetry or DB values to float without ever raising.

    NULL, empty strings, placeholder text and unparseable junk all return `default`,
    so one bad daemon write cannot take down the whole monitoring page.
    """
    try:
        if val is None:
            return default
        if isinstance(val, str):
            if val.strip().lower() in _JUNK_TOKENS:
                return default
            val = float(val.replace(",", "").replace("$", "").strip())
        else:
            val = float(val)
        if val != val:  # NaN
            return default
        return val
    except (TypeError, ValueError):
        return default


def state_num(key: str, default: float = 0.0) -> float:
    """Read a numeric system_state value without ever raising."""
    return num(query_system_state(key, ""), default)


def state_flag(key: str, default: str = "0") -> bool:
    """Parse a 0/1-style operator or daemon halt flag, case and whitespace insensitive.

    A control the fleet honours but the HUD misreads as OFF is worse than no HUD at all.
    """
    return str(query_system_state(key, default)).strip().lower() in ("1", "true", "yes", "on")


def read_state_now(key: str):
    """Uncached single read of system_state, used to verify operator writes landed."""
    try:
        conn = connect_readonly(TRADING_DATA_DB)
        try:
            row = conn.execute("SELECT value FROM system_state WHERE key = ?", (key,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None
    except Exception:
        return None


def state_flag_now(key: str) -> bool:
    """Parse a halt flag straight from disk, bypassing the 4s telemetry cache.

    Operator controls must compare against on-disk truth: a cached read here makes the
    widget disagree with the database and re-issue the same write on every rerun.
    """
    val = read_state_now(key)
    return str(val if val is not None else "0").strip().lower() in ("1", "true", "yes", "on")


def execute_safe_write(db_path: str, query: str, params: tuple = ()) -> bool:
    """Executes a non-blocking parameterized write to SQLite with timeout."""
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute(query, params)
        conn.commit()
        return True
    except Exception as w_err:
        FEED_FAILURES[f"write \u2192 {os.path.basename(db_path)}"] = f"{type(w_err).__name__}: {str(w_err)[:120]}"
        return False
    finally:
        if conn is not None:
            conn.close()

def utc_to_sast(dt_val) -> str:
    """Converts a UTC datetime string/object to South Africa Standard Time (SAST / UTC+2)."""
    if dt_val is None or str(dt_val).strip() in ("", "None", "N/A"):
        return "N/A"
    try:
        if isinstance(dt_val, str):
            clean_str = dt_val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
        else:
            dt = dt_val
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        sast_tz = pytz.timezone("Africa/Johannesburg")
        return dt.astimezone(sast_tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_val)[:19]

def compute_position_age(entry_time_val) -> str:
    """Computes human-readable position holding time from entry timestamp."""
    if not entry_time_val or str(entry_time_val).strip() in ("", "None", "N/A"):
        return "Active"
    try:
        clean_str = str(entry_time_val).replace("Z", "+00:00")
        dt_entry = datetime.fromisoformat(clean_str)
        if dt_entry.tzinfo is None:
            dt_entry = pytz.utc.localize(dt_entry)
        now_utc = datetime.now(timezone.utc)
        elapsed_secs = max(0, int((now_utc - dt_entry).total_seconds()))
        mins = elapsed_secs // 60
        secs = elapsed_secs % 60
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            return f"{hours}h {mins:02d}m"
        return f"{mins}m {secs:02d}s"
    except Exception:
        return "Active"

def calculate_estimated_liquidation(entry_p: float, lev: float, side: str, mmr: float = 0.005) -> float:
    """Calculates estimated isolated liquidation price given leverage and MMR buffer."""
    if entry_p <= 0 or lev <= 0:
        return 0.0
    if side == "long":
        return max(0.0, entry_p * (1.0 - (1.0 / lev) + mmr))
    else:
        return entry_p * (1.0 + (1.0 / lev) - mmr)

def calculate_liq_distance_pct(curr_p: float, liq_p: float, side: str) -> float:
    """Calculates distance to liquidation as a percentage of current mark price."""
    if curr_p <= 0 or liq_p <= 0:
        return 0.0
    if side == "long":
        return max(0.0, ((curr_p - liq_p) / curr_p) * 100.0)
    else:
        return max(0.0, ((liq_p - curr_p) / curr_p) * 100.0)


# ── Fleet active-position deck: one measured-or-absent reading per row ──────────────────
# The deck renders five engine tables as one frame and then computes risk figures off it. It
# used to invent every input it could not read — a ±3.5% stop the row never carried, printed as
# "(Exchange)" — and each invention was re-typed in the aggregation loop, the card loop and the
# chart-tab loop. One row now produces one reading, shared by all three, and an unmeasured input
# publishes an absence instead of a number.
_DECK_ABSENT_TEXT = "not measured"
_DECK_ABSENT_COLOR = "#94a3b8"
# The maintenance margin the deck's liquidation model assumes. No producer publishes an MMR and no
# column stores one, so every figure built on it is labelled as the panel's own model where it shows.
_PANEL_LIQ_MODEL_MMR = 0.005


def _deck_absence(reason: str) -> dict:
    """A figure the deck could not read: no value, and the reason it says on screen."""
    return {"value": None, "measured": False, "reason": str(reason)}


def _deck_reading(value: float) -> dict:
    """A figure the deck did read. A recorded zero is a fact and stays a reading."""
    return {"value": float(value), "measured": True, "reason": ""}


def _deck_is_absent(read) -> bool:
    return not isinstance(read, dict) or not read.get("measured") or read.get("value") is None


def _deck_text(read, fmt: str = "{:,.4f}") -> str:
    """One reading's on-screen value, or the house absence."""
    if _deck_is_absent(read):
        return _DECK_ABSENT_TEXT
    try:
        return fmt.format(read["value"])
    except (TypeError, ValueError):
        return _DECK_ABSENT_TEXT


def _deck_note(read) -> str:
    """The dimmed reason beside an absent figure; nothing beside a measured one."""
    return "" if not _deck_is_absent(read) else _absence_note(read.get("reason", ""))


def _deck_color(read, measured_color: str) -> str:
    """A measured figure keeps its colour; an absence is grey — never an all-clear, never an alarm."""
    return _DECK_ABSENT_COLOR if _deck_is_absent(read) else measured_color


def _deck_chart_level(read) -> float:
    """A measured level for the candle chart, or 0.0 — its own 'draw nothing' encoding."""
    return _deck_value_or(read, 0.0)


def _deck_value_or(read, fallback: float = 0.0) -> float:
    """The measured number a layout needs, or the fallback that shows nothing."""
    return fallback if _deck_is_absent(read) else float(read["value"])


def _deck_account_balance(live_bal_val) -> dict:
    """The equity the utilization gauge divides by.

    `live_bal_val` is None until the balance syncs; the pre-fix gauge substituted 0.0 and printed a
    0.0% utilization in the all-clear green for an account it cannot see.
    """
    value = _optional_float(live_bal_val)
    if value is None or value <= 0.0:
        return _deck_absence("no account balance synced")
    return _deck_reading(value)


def _deck_margin_gauge(margin_locked: dict, balance: dict, limit: dict) -> dict:
    """Utilization, bar width and free buffer as readings: absent if any input is absent.

    A ratio whose numerator came from an assumed leverage is not a utilization, and a buffer
    subtracted from a balance that was never read is not a buffer.
    """
    for read, name in ((margin_locked, "margin total"), (balance, "account balance"),
                       (limit, "margin limit")):
        if _deck_is_absent(read):
            return {"utilization": _deck_absence(f"no utilization without a measured {name}"),
                    "bar_width": _deck_absence(f"no gauge without a measured {name}"),
                    "buffer": _deck_absence(f"no buffer without a measured {name}")}
    util = margin_locked["value"] / balance["value"] * 100.0
    return {
        "utilization": _deck_reading(util),
        "bar_width": _deck_reading(min(100.0, (util / limit["value"]) * 100.0)),
        "buffer": _deck_reading(max(0.0, balance["value"] - margin_locked["value"])),
    }


def _deck_margin_rung(util: dict) -> dict:
    """Colour and status words for one utilization figure — grey and no verdict for an absence."""
    if _deck_is_absent(util):
        return {"color": _DECK_ABSENT_COLOR, "status": "NOT MEASURED"}
    value = util["value"]
    if value < 45.0:
        return {"color": "#00ff66", "status": "SAFE BUFFER"}
    if value < 65.0:
        return {"color": "#ff9900", "status": "ELEVATED EXPOSURE"}
    return {"color": "#ff003c", "status": "CRITICAL / NEARING CEILING"}


def _position_hard_sl(row) -> dict:
    """The hard stop the row records, or an absence — never a price the panel estimated.

    `open_positions` carries a `hard_sl` column and the reconciliation recovery path computes one
    (reconciliation_engine.py:811-842); `swing_open_positions` has no such column at all
    (verified with PRAGMA table_info), so after `pd.concat` every swing row reads NaN, `num()`
    turns that into 0.0 and the deck's ±3.5% stand-in rendered a panel-invented price under the
    label "(Exchange)" for every position the fleet can open today.

    A 0.0 or negative here is an absence, not a reading: a stop is a price and this column's DDL
    default is 0.0, so zero is "the engine that opened this position never recorded a stop".
    """
    value = _optional_float(row.get("hard_sl") if hasattr(row, "get") else None)
    if value is None:
        return _deck_absence("no hard SL recorded for this position")
    if value <= 0.0:
        return _deck_absence("recorded hard SL is not a price")
    return _deck_reading(value)


def _position_leverage(row) -> dict:
    """The leverage the row records, or an absence — never the 10x/5x the panel assumed.

    `leverage` is a real measured column: `reconciliation_engine.py:830` writes the exchange's own
    value, `super_bot.py:1282-1289` writes `self.grid_leverage` (config.GRID_LEVERAGE) and
    `swing_trader.py:2116-2118` writes `pos_record["leverage"]`. The deck's default
    (`10.0 if "BTC" else 5.0`) applied wherever the column was NULL or missing, which is a live
    path — no INSERT exists anywhere for `berserker_open_positions`, `marl_live_open_positions` or
    `mean_reversion_open_positions` — and 10x is double the highest leverage the repo's own config
    defaults at (`config.py:142` 5, `config.py:269-274` 5; `.env` may raise either), so it halved
    the margin it printed.

    A 0 or negative leverage is an absence too: it is not a leverage the exchange can open a
    position at, and dividing margin by it is undefined. The `or` this replaces treated a measured
    0.0 exactly like a missing value and printed the assumed 10x for it.
    """
    value = _optional_float(row.get("leverage") if hasattr(row, "get") else None)
    if value is None:
        return _deck_absence("no leverage recorded for this position")
    if value <= 0.0:
        return _deck_absence("recorded leverage is not a margin denominator")
    return _deck_reading(value)


def _position_margin(qty: dict, entry_p: dict, lev: dict) -> dict:
    """Notional divided by the recorded leverage. Absent if any input is absent (C-2).

    The pre-fix `if lev > 0 else (qty * entry_p)` published the full notional as the margin for a
    row with no leverage — an implied 1x nobody measured.
    """
    for read, name in ((qty, "size"), (entry_p, "entry price"), (lev, "leverage")):
        if _deck_is_absent(read):
            return _deck_absence(f"no margin without a measured {name}")
    return _deck_reading((qty["value"] * entry_p["value"]) / lev["value"])


def _position_liquidation(entry_p: dict, lev: dict, side: str) -> dict:
    """The panel's OWN liquidation estimate, and an absence unless both inputs were read.

    `calculate_estimated_liquidation`'s `mmr` is a hardcoded 0.5% maintenance margin: nothing in
    `system_state`, no column and no producer supplies one (the only MMR ever written anywhere was
    0.004 at archive/experimental_traders/basis_trader.py:279, in dead code). So even a liquidation
    price built from a measured leverage is a model figure, not a reading, and is labelled as the
    panel's own everywhere it appears. With an absent leverage the deck prints nothing at all: a
    liquidation price derived from an assumed leverage is not a risk number an operator can act on.
    """
    if _deck_is_absent(entry_p):
        return _deck_absence("no liquidation estimate without a measured entry price")
    if _deck_is_absent(lev):
        return _deck_absence("no liquidation estimate without a measured leverage")
    price = calculate_estimated_liquidation(entry_p["value"], lev["value"], side,
                                            mmr=_PANEL_LIQ_MODEL_MMR)
    if price <= 0.0:
        return _deck_absence("the panel's liquidation model produced no price")
    return _deck_reading(price)


def _position_roi(pnl: dict, base: dict) -> dict:
    """Return on the locked margin. An absent input used to print a measured-looking 0.00%.

    The pre-fix `(pnl / margin * 100.0) if margin > 0 else 0.0` fell into its 0.0 branch for
    exactly the rows whose margin was invented, so the deck reported a flat return on a position
    it could not size. C-2 supplies the absent `base`; C-3 supplies the absent `pnl`.
    """
    for read, name in ((pnl, "PnL"), (base, "locked margin")):
        if _deck_is_absent(read):
            return _deck_absence(f"no return on an unmeasured {name}")
    if base["value"] <= 0.0:
        return _deck_absence("no return on a margin that is not a denominator")
    return _deck_reading(pnl["value"] / base["value"] * 100.0)


def _position_liq_buffer(curr_p: dict, est_liq: dict, side: str) -> dict:
    """Distance to the panel's liquidation model, or an absence saying which input is missing.

    `calculate_liq_distance_pct` returns 0.0 for an unmeasured price, which the deck printed as
    "+0.0% safety" beside a liquidation price it had invented off an assumed leverage.
    """
    for read, name in ((curr_p, "live price"), (est_liq, "liquidation estimate")):
        if _deck_is_absent(read):
            return _deck_absence(f"no safety buffer without a measured {name}")
    distance = calculate_liq_distance_pct(curr_p["value"], est_liq["value"], side)
    if distance <= 0.0:
        return _deck_absence("the panel's liquidation model gives no safety distance")
    return _deck_reading(distance)


def _position_mark(live_prices, symbol: str, clean_symbol: str = "") -> dict:
    """The feed's own quote for this position, or an absence — never the entry price wearing it.

    `fetch_live_market_prices` swallows both of its failures, so a symbol can be absent from the
    dict entirely (a missing key is information, contract item 8). The deck filled that gap with
    the entry price, which made every unmeasured position print a flat mark under "Mark" and
    under the column "Live Price" — authority labels for a source the panel did not read.
    """
    table = live_prices if isinstance(live_prices, dict) else {}
    raw = str(symbol)
    base = raw.split(":")[0]
    keys = []
    for candidate in (clean_symbol, base.upper().replace("/", "").replace("-", ""), raw, base):
        if candidate and candidate not in keys:
            keys.append(candidate)
    for key in keys:
        if key in table:
            value = _optional_float(table.get(key))
            if value is None:
                return _deck_absence(f"the price feed's quote for {key} is not a number")
            if value <= 0.0:
                return _deck_absence(f"the price feed's quote for {key} is not a price")
            return _deck_reading(value)
    return _deck_absence("no live price for this symbol in the current fetch")


def _position_funding(funding_map, symbol: str, clean_symbol: str = "") -> dict:
    """The accrued funding the telemetry reports for this position, or an absence — never a verdict.

    `fetch_live_position_funding` returns {} when `system_state` carries no `live_position_funding`
    row and the Bitget UTA fallback fails, which is every symbol today (contract item 9). The deck
    read that empty map through `num(..., 0.0)`, so "nothing reported" became a recorded zero,
    printed "(Neutral)" beside it and was added to PnL. A funding of exactly 0.0 IS a measurement —
    the position neither paid nor received — and stays a reading.
    """
    table = funding_map if isinstance(funding_map, dict) else {}
    raw = str(symbol)
    base = raw.split(":")[0]
    keys = []
    for candidate in (clean_symbol, base.upper().replace("/", "").replace("-", ""), raw, base):
        if candidate and candidate not in keys:
            keys.append(candidate)
    for key in keys:
        info = table.get(key)
        if not isinstance(info, dict) or "total_funding" not in info:
            continue
        value = _optional_float(info.get("total_funding"))
        if value is None:
            return _deck_absence(f"the funding telemetry's total_funding for {key} is not a number")
        return _deck_reading(value)
    return _deck_absence("no funding telemetry for this symbol")


def _position_unrealized_pnl(curr_p: dict, entry_p: dict, qty: dict, side: str) -> dict:
    """Mark-to-market PnL: publishable only when the mark, the entry and the size were all read.

    The pre-fix arithmetic produced exactly 0.0 whenever the feed had no quote, because the
    stand-in mark WAS the entry price — and the deck then coloured that flat number green.
    """
    for read, name in ((curr_p, "live price"), (entry_p, "entry price"), (qty, "size")):
        if _deck_is_absent(read):
            return _deck_absence(f"no unrealized PnL without a measured {name}")
    delta = (curr_p["value"] - entry_p["value"] if side == "long"
             else entry_p["value"] - curr_p["value"])
    return _deck_reading(delta * qty["value"])


def _deck_pnl_color(read) -> str:
    """Green or red only for a measured PnL; grey for an absence, which is never an all-clear."""
    if _deck_is_absent(read):
        return _DECK_ABSENT_COLOR
    return "#00ff66" if read["value"] >= 0 else "#ff003c"


def _deck_signed_money(read, digits: int = 4) -> str:
    """`+$1.2345` for a measured figure and the house absence for one that was never taken."""
    if _deck_is_absent(read):
        return _DECK_ABSENT_TEXT
    value = read["value"]
    return f"{'+' if value >= 0 else '-'}${abs(value):,.{digits}f}"


def _position_level_distance(curr_p: dict, level: dict, side: str, kind: str) -> dict:
    """Percent from the live price to a TP/SL level, absent when either input is.

    These read 0.00% for a position with no quote, because the price they divided by was the
    entry price the panel had substituted.
    """
    for read, name in ((curr_p, "live price"), (level, f"{kind} level")):
        if _deck_is_absent(read):
            return _deck_absence(f"no distance to {kind} without a measured {name}")
    if curr_p["value"] <= 0.0:
        return _deck_absence("no distance on a live price that is not a denominator")
    price = curr_p["value"]
    target = level["value"]
    if kind == "tp":
        gap = target - price if side == "long" else price - target
    else:
        gap = price - target if side == "long" else target - price
    return _deck_reading(gap / price * 100.0)


def _position_target_label(dist: dict, side: str, kind: str) -> str:
    """The card's target wording, or an absence naming its own field."""
    if _deck_is_absent(dist):
        return f"{_DECK_ABSENT_TEXT} ({kind})"
    value = dist["value"]
    if kind == "tp":
        return f"{value:+.2f}% to TP" if side == "long" else f"-{value:.2f}% to TP"
    return f"{value:.2f}% above SL" if side == "long" else f"{value:.2f}% below SL"


def _deck_sum(*reads: dict) -> dict:
    """A published sum is a measurement only when every addend is one, and says which is not."""
    values = []
    for read in reads:
        if _deck_is_absent(read):
            reason = read.get("reason", "") if isinstance(read, dict) else ""
            return _deck_absence(f"no total — {reason}" if reason else "no measured addend to total")
        values.append(float(read["value"]))
    return _deck_reading(sum(values)) if values else _deck_absence("nothing to total")


def _deck_funding_word(read) -> str:
    """Gained, Paid or Flat for a funding figure the deck read; no verdict at all for one it did not."""
    if _deck_is_absent(read):
        return _DECK_ABSENT_TEXT
    value = read["value"]
    return "Gained" if value > 0.0 else ("Paid" if value < 0.0 else "Flat")


def _deck_funding_color(read) -> str:
    """Green or orange only for a measured funding figure.

    The pre-fix badge compared the 0.0 it had substituted for telemetry against zero, so an unread
    funding came back in the all-clear green.
    """
    if _deck_is_absent(read):
        return _DECK_ABSENT_COLOR
    return "#00ff66" if read["value"] >= 0.0 else "#ffaa00"


def _deck_funding_text(read) -> str:
    """`+$0.4210 (Gained)` for a measured funding figure; the house absence for an unread one."""
    if _deck_is_absent(read):
        return _DECK_ABSENT_TEXT
    return f"{_deck_signed_money(read)} ({_deck_funding_word(read)})"


def _deck_funding_total(reads) -> dict:
    """The fleet's accrued funding: a total only when every position reported one (C-4).

    `total_live_funding` was a float initialised to 0.0 that each row added its funding stand-in to,
    so a book with no telemetry summed to a printed figure. Summing only the measured rows would be
    the same claim with a smaller number behind it, so the total says who is missing.
    """
    readings = list(reads)
    if not readings:
        return _deck_absence("no open positions to total funding")
    missing = [read for read in readings if _deck_is_absent(read)]
    if missing:
        return _deck_absence(f"{len(missing)} of {len(readings)} positions report no funding")
    return _deck_reading(sum(float(read["value"]) for read in readings))


def _position_deck_readings(row, live_prices=None, funding_map=None) -> dict:
    """Every figure the deck publishes for one open position row, measured or absent.

    Shared by the aggregation loop, the card loop and the chart-tab loop, which each used to
    re-derive the same numbers — and re-type the same inventions.
    """
    symbol = str(row["symbol"]) if hasattr(row, "__getitem__") else str(row.get("symbol", ""))
    clean_symbol = symbol.upper().replace("/", "").replace("-", "")
    side = str(row.get("side") or "long").lower()

    def _column(key: str) -> dict:
        value = _optional_float(row.get(key) if hasattr(row, "get") else None)
        return _deck_absence(f"no {key} recorded for this position") if value is None \
            else _deck_reading(value)

    qty = _column("qty")
    entry_p = _column("entry_price")
    lev = _position_leverage(row)
    margin = _position_margin(qty, entry_p, lev)
    hard_sl = _position_hard_sl(row)
    est_liq = _position_liquidation(entry_p, lev, side)
    mark = _position_mark(live_prices, symbol, clean_symbol)
    funding = _position_funding(funding_map, symbol, clean_symbol)
    tp = _column("take_profit")
    sl = _column("stop_loss")
    unr_pnl = _position_unrealized_pnl(mark, entry_p, qty, side)
    return {
        "symbol": symbol,
        "clean_symbol": clean_symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_p,
        "stop_loss": sl,
        "take_profit": tp,
        "leverage": lev,
        "margin": margin,
        "hard_sl": hard_sl,
        "est_liq": est_liq,
        "mark": mark,
        "unr_pnl": unr_pnl,
        "funding": funding,
        "unr_roi": _position_roi(unr_pnl, margin),
        "dist_liq": _position_liq_buffer(mark, est_liq, side),
        "dist_tp": _position_level_distance(mark, tp, side, "tp"),
        "dist_sl": _position_level_distance(mark, sl, side, "sl"),
    }


def _fleet_position_rollup(positions) -> dict:
    """The deck's portfolio totals, each one a measurement only when every addend was one.

    `total_margin_locked` fed the utilization gauge, and an absent leverage made the addend a
    guess — at 10x, half of the margin the gauge then reported. `total_unrealized_pnl` fed the
    strip and the green-at-zero rule, and its addends were marks the deck had substituted from the
    entry price. Summing only the measured rows would fix neither: a partial sum published as a
    total is the same claim with a smaller number behind it, so each total says what is missing
    and how many rows are missing it.
    """
    counts = len(positions)
    rollup = {"count": counts}
    for key, total_key, noun in (("margin", "margin_locked", "margin"),
                                 ("unr_pnl", "unrealized_pnl", "mark PnL")):
        missing = [p for p in positions if _deck_is_absent(p.get(key))]
        if not positions:
            rollup[total_key] = _deck_absence(f"no open positions to total {noun}")
            rollup[f"{key}_unmeasured"] = 0
        elif missing:
            rollup[total_key] = _deck_absence(
                f"{len(missing)} of {counts} positions report no {noun}")
            rollup[f"{key}_unmeasured"] = len(missing)
        else:
            rollup[total_key] = _deck_reading(
                sum(p[key]["value"] for p in positions))
            rollup[f"{key}_unmeasured"] = 0
    return rollup


def _deck_label(read, name: str, fmt: str = "{:,.4f}") -> str:
    """`5x`, or `leverage not measured`: an absence names its own field so the card stays readable."""
    return _deck_text(read, fmt) if not _deck_is_absent(read) else f"{name} {_DECK_ABSENT_TEXT}"


def get_local_ip() -> str:
    """Detects local LAN IP for Wi-Fi mobile browsing."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# ── 5.1 Real-Time Live Price Fetcher (Local Fast Cache + Batch API) ──────────
@st.cache_data(ttl=5)
def fetch_live_market_prices() -> dict:
    """Fetches real-time market prices across assets with local JSON cache fallback."""
    prices = {}
    # 1. Check local sub-millisecond microstructure state
    state_file = MICROSTRUCTURE_STATE_FILE
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f).get("symbols", {})
                for s, d in data.items():
                    bid = float(d.get("best_bid", 0.0) or 0.0)
                    ask = float(d.get("best_ask", 0.0) or 0.0)
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                        clean_s = s.replace("/", "").replace("-", "").upper()
                        prices[clean_s] = mid
                        prices[s] = mid
        except Exception:
            pass

    # 2. Batch Bitget USDT-FUTURES Tickers API (native exchange coverage)
    try:
        resp = requests.get("https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES", timeout=1.5)
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                sym = item.get("symbol", "")
                p = float(item.get("lastPr", 0.0) or 0.0)
                if sym and p > 0:
                    prices[sym] = p
                    if sym.endswith("USDT"):
                        base = sym[:-4]
                        prices[f"{base}/USDT"] = p
    except Exception:
        pass
    return prices

@st.cache_data(ttl=5)
def fetch_live_position_funding() -> dict:
    """Fetches real-time position funding telemetry from system_state, with fast fallback to Bitget UTA."""
    raw = query_system_state("live_position_funding", "")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    # Fallback to direct query if system_state is uninitialized
    try:
        from bitget_uta_client import BitgetUTAClient
        import asyncio

        async def _fetch():
            c = BitgetUTAClient()
            try:
                pos = await c.fetch_positions()
                return pos
            finally:
                await c.exchange.close()

        def _run_sync(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result(timeout=4.0)
            else:
                return asyncio.run(coro)

        positions = _run_sync(_fetch())
        res = {}
        for p in positions:
            sym_raw = p.get("symbol", "")
            total_qty = float(p.get("total", 0.0) or p.get("contracts", 0.0) or 0.0)
            if total_qty > 0 and sym_raw:
                t_fund = float(p.get("totalFunding", 0.0) or 0.0)
                info = {
                    "total_funding": t_fund,
                    "open_fee_total": float(p.get("openFeeTotal", 0.0) or 0.0),
                    "close_fee_total": float(p.get("closeFeeTotal", 0.0) or 0.0),
                    "cur_realised_pnl": float(p.get("curRealisedPnl", 0.0) or 0.0),
                    "unrealised_pnl": float(p.get("unrealisedPnl", 0.0) or 0.0),
                }
                res[sym_raw] = info
                sym_clean = sym_raw.replace("USDT", "/USDT") if "/" not in sym_raw and sym_raw.endswith("USDT") else sym_raw
                res[sym_clean] = info
                res[sym_raw.upper().replace("/", "").replace("-", "")] = info
        return res
    except Exception:
        return {}

@st.cache_data(ttl=15)
def fetch_position_candles(symbol: str, timeframe: str = "15m", limit: int = 50) -> pd.DataFrame:
    """Fetches real-time OHLCV candlestick data for an active position with Bitget & Binance dual fallback."""
    clean_sym = symbol.upper().replace("/", "").replace("-", "").split(":")[0]
    if not clean_sym.endswith("USDT"):
        clean_sym += "USDT"
    
    # 1. Primary: Bitget USDT-FUTURES Mix Candles
    bg_tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "1H": "1H", "4h": "4H", "4H": "4H", "1d": "1D", "1D": "1D"
    }
    bg_gran = bg_tf_map.get(timeframe, "15m")
    try:
        url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={clean_sym}&granularity={bg_gran}&limit={limit}&productType=USDT-FUTURES"
        resp = requests.get(url, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
                df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
                for col in ["open", "high", "low", "close", "base_vol"]:
                    df[col] = df[col].astype(float)
                return df
    except Exception:
        pass

    # 2. Secondary Failover: Binance Public Klines
    bn_tf = timeframe.lower()
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={clean_sym}&interval={bn_tf}&limit={limit}"
        resp = requests.get(url, timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list):
                df = pd.DataFrame(data, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "tbba", "tbqa", "ignore"
                ])
                df["ts"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
                df.rename(columns={"volume": "base_vol"}, inplace=True)
                for col in ["open", "high", "low", "close", "base_vol"]:
                    df[col] = df[col].astype(float)
                return df[["ts", "open", "high", "low", "close", "base_vol"]]
    except Exception:
        pass

    return pd.DataFrame()

def build_position_candlestick_chart(
    symbol: str,
    side: str = "long",
    entry_p: float = 0.0,
    sl: float = 0.0,
    hard_sl: float = 0.0,
    tp: float = 0.0,
    curr_p: float = 0.0,
    est_liq: float = 0.0,
    timeframe: str = "15m",
    limit: int = 50,
    show_emas: bool = True,
    show_vp: bool = True
) -> Optional[go.Figure]:
    """Generates an interactive candlestick and volume chart with execution levels & Go-Stock quant overlays."""
    df = fetch_position_candles(symbol, timeframe=timeframe, limit=limit)
    if df is None or df.empty or len(df) < 5:
        return None

    # Calculate EMAs if requested
    if show_emas:
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    # Calculate TD Sequential setup count
    td_subtitle = ""
    try:
        df_td = df.copy()
        td_data = compute_td_sequential(df_td)
        td_cnt = td_data.get("count", 0)
        if td_data.get("is_td9_upward_exhaustion"):
            td_subtitle = " | ⚠️ TD-9 BUY CLIMAX (Exhaustion Top)"
        elif td_data.get("is_td9_downward_exhaustion"):
            td_subtitle = " | 🟢 TD-9 SELL CLIMAX (Exhaustion Bottom)"
        elif td_cnt != 0:
            td_subtitle = f" | TD Setup: {td_cnt:+d}"
    except Exception:
        pass

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.80, 0.20]
    )

    # 1. Candlestick Trace
    fig.add_trace(go.Candlestick(
        x=df["ts"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=f"{symbol} Price",
        increasing_line_color="#00ff66",
        decreasing_line_color="#ff003c",
        increasing_fillcolor="rgba(0, 255, 102, 0.25)",
        decreasing_fillcolor="rgba(255, 0, 60, 0.25)",
        hoverinfo="x+y+text"
    ), row=1, col=1)

    # 2. EMAs
    if show_emas:
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["ema9"],
            line=dict(color="#facc15", width=1.3),
            name="EMA 9 (Fast)"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df["ts"], y=df["ema21"],
            line=dict(color="#06b6d4", width=1.3),
            name="EMA 21 (Trend)"
        ), row=1, col=1)

    # 3. Volume Subplot
    vol_colors = ["#00ff66" if c >= o else "#ff003c" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["ts"],
        y=df["base_vol"],
        marker_color=vol_colors,
        name="Volume",
        opacity=0.45,
        hoverinfo="y"
    ), row=2, col=1)

    # 4. Overlays: Volume Profile 70% Value Area Levels (POC, VAH, VAL)
    if show_vp and len(df) >= 15:
        try:
            df_vp = df.copy()
            df_vp["volume"] = df_vp["base_vol"]
            vp_data = compute_volume_profile(df_vp, lookback=min(len(df), 48), value_area_pct=0.70)
            poc_val = vp_data.get("poc")
            vah_val = vp_data.get("vah")
            val_val = vp_data.get("val")

            if poc_val and poc_val > 0:
                fig.add_hline(
                    y=poc_val, line_dash="dash", line_color="#38bdf8", line_width=1.3,
                    annotation_text=f"POC: ${poc_val:,.4f}",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#38bdf8", size=10, family="JetBrains Mono"),
                    row=1, col=1
                )
            if vah_val and vah_val > 0:
                fig.add_hline(
                    y=vah_val, line_dash="dot", line_color="#c084fc", line_width=1.1,
                    annotation_text=f"VAH: ${vah_val:,.4f}",
                    annotation_position="top right",
                    annotation_font=dict(color="#c084fc", size=9, family="JetBrains Mono"),
                    row=1, col=1
                )
            if val_val and val_val > 0:
                fig.add_hline(
                    y=val_val, line_dash="dot", line_color="#c084fc", line_width=1.1,
                    annotation_text=f"VAL: ${val_val:,.4f}",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#c084fc", size=9, family="JetBrains Mono"),
                    row=1, col=1
                )
        except Exception:
            pass

    # 5. Overlays: Entry Price
    if entry_p > 0:
        fig.add_hline(
            y=entry_p, line_dash="dash", line_color="#00e5ff", line_width=1.8,
            annotation_text=f"Entry: ${entry_p:,.4f}",
            annotation_position="top left",
            annotation_font=dict(color="#00e5ff", size=11, family="JetBrains Mono"),
            row=1, col=1
        )

    # 6. Overlays: Soft Stop Loss
    if sl > 0:
        sl_pct = abs(entry_p - sl) / entry_p * 100.0 if entry_p > 0 else 0.0
        fig.add_hline(
            y=sl, line_dash="dash", line_color="#ffaa00", line_width=1.6,
            annotation_text=f"Soft SL: ${sl:,.4f} (-{sl_pct:.2f}%)",
            annotation_position="bottom left",
            annotation_font=dict(color="#ffaa00", size=11, family="JetBrains Mono"),
            row=1, col=1
        )

    # 7. Overlays: Hard SL — only a recorded level reaches this; the caller passes 0.0 for an absence.
    if hard_sl > 0 and abs(hard_sl - sl) > 0.0001:
        fig.add_hline(
            y=hard_sl, line_dash="dot", line_color="#ff003c", line_width=1.4,
            annotation_text=f"Hard SL: ${hard_sl:,.4f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#ff003c", size=10, family="JetBrains Mono"),
            row=1, col=1
        )

    # 8. Overlays: Take Profit
    if tp > 0:
        tp_pct = abs(tp - entry_p) / entry_p * 100.0 if entry_p > 0 else 0.0
        fig.add_hline(
            y=tp, line_dash="dash", line_color="#00ff66", line_width=1.6,
            annotation_text=f"Take Profit: ${tp:,.4f} (+{tp_pct:.2f}%)",
            annotation_position="top left",
            annotation_font=dict(color="#00ff66", size=11, family="JetBrains Mono"),
            row=1, col=1
        )

    # 9. Overlays: Current Mark Price
    mark_price = curr_p if curr_p > 0 else (num(df["close"].iloc[-1], 0.0) if not df.empty else 0.0)
    if mark_price > 0:
        curr_is_profit = ((side == "long" and mark_price >= entry_p) or (side == "short" and mark_price <= entry_p)) if entry_p > 0 else True
        curr_color = "#00ff66" if curr_is_profit else "#ff003c"
        fig.add_hline(
            y=mark_price, line_dash="dot", line_color=curr_color, line_width=1.3,
            annotation_text=f"Live Mark: ${mark_price:,.4f}",
            annotation_position="top right",
            annotation_font=dict(color=curr_color, size=11, family="JetBrains Mono"),
            row=1, col=1
        )

    # 10. Overlays: Estimated Liquidation Price (if visible near price range)
    if est_liq > 0:
        min_p = df["low"].min()
        max_p = df["high"].max()
        if min_p * 0.75 <= est_liq <= max_p * 1.25:
            fig.add_hline(
                y=est_liq, line_dash="dot", line_color="#7f1d1d", line_width=1.2,
                annotation_text=f"Est Liq: ${est_liq:,.4f}",
                annotation_position="bottom right",
                annotation_font=dict(color="#f87171", size=10, family="JetBrains Mono"),
                row=1, col=1
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0d121f",
        height=480,
        margin=dict(l=15, r=15, t=35, b=15),
        title=dict(
            text=f"<b>{symbol}</b> — Timeframe: {timeframe.upper()}{td_subtitle}",
            font=dict(size=13, color="#f8fafc", family="JetBrains Mono"),
            x=0.01, y=0.98
        ),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#94a3b8")
        ),
        yaxis=dict(title="Price (USDT)", gridcolor="#1e293b", color="#94a3b8"),
        yaxis2=dict(title="Vol", gridcolor="#1e293b", color="#64748b"),
        xaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
        xaxis2=dict(gridcolor="#1e293b", color="#64748b"),
    )
    return fig

@st.cache_data(ttl=45)
def fetch_fleet_quant_alpha_matrix(symbols: tuple, timeframe: str = "4h") -> List[Dict[str, Any]]:
    """Calculates Go-Stock quantitative alpha gates (Divergence, Volume Profile, TD-Sequential) across fleet."""
    matrix = []
    for sym in symbols:
        try:
            df = fetch_position_candles(sym, timeframe=timeframe, limit=50)
            if df is None or df.empty or len(df) < 20:
                continue
            df_calc = df.copy()
            df_calc["volume"] = df_calc["base_vol"]
            
            curr_p = num(df_calc["close"].iloc[-1], 0.0)
            div_info = detect_momentum_divergence(df_calc, lookback=24)
            vp_info = compute_volume_profile(df_calc, lookback=min(len(df_calc), 48), value_area_pct=0.70)
            td_info = compute_td_sequential(df_calc)
            
            # Divergence verdict
            if div_info.get("bearish_divergence"):
                div_str = "🔴 Bearish Div (Bull-Trap)"
                div_verdict = "VETO_LONG"
            elif div_info.get("bullish_divergence"):
                div_str = "🟢 Bullish Div (Bear-Trap)"
                div_verdict = "FAVOR_LONG"
            else:
                div_str = "⚪ Clean (No Divergence)"
                div_verdict = "NEUTRAL"

            # Volume Profile zone
            poc = vp_info.get("poc", curr_p)
            vah = vp_info.get("vah", curr_p)
            val = vp_info.get("val", curr_p)
            if curr_p >= (vah * 0.998):
                vp_zone = f"🚀 Above VAH (${vah:,.2f}) [Breakout]"
                vp_pass = True
            elif curr_p <= (val * 1.002):
                vp_zone = f"📉 Below VAL (${val:,.2f}) [Breakdown]"
                vp_pass = False
            else:
                vp_zone = f"📦 In Value Area (${val:,.2f}-${vah:,.2f}) [Chop]"
                vp_pass = False

            # TD Sequential
            td_cnt = td_info.get("count", 0)
            if td_info.get("is_td9_upward_exhaustion"):
                td_str = f"⚠️ TD-9 BUY CLIMAX ({td_cnt})"
                td_pass = False
            elif td_info.get("is_td9_downward_exhaustion"):
                td_str = f"🟢 TD-9 SELL CLIMAX ({td_cnt})"
                td_pass = True
            elif td_cnt > 0:
                td_str = f"Setup +{td_cnt} (Buy Run)"
                td_pass = True
            elif td_cnt < 0:
                td_str = f"Setup {td_cnt} (Sell Run)"
                td_pass = True
            else:
                td_str = "Setup 0 (Neutral)"
                td_pass = True

            # Overall Gate Clearance Verdict
            if div_verdict == "VETO_LONG":
                gate_clearance = "🚫 VETO (Bearish Divergence)"
            elif not vp_pass:
                gate_clearance = "🚫 VETO (Trapped in Value Area)"
            elif not td_pass:
                gate_clearance = "🚫 VETO (TD-9 Exhaustion Top)"
            else:
                gate_clearance = "✅ PASS (All Gates Cleared)"

            matrix.append({
                "Asset": sym,
                "Live Price": f"${curr_p:,.4f}",
                "4H Divergence": div_str,
                "Volume Profile Zone": vp_zone,
                "POC Level": f"${poc:,.2f}",
                "TD-Sequential": td_str,
                "Gate Clearance": gate_clearance
            })
        except Exception:
            pass
    return matrix

_LOG_TALLY_KEYS = ("gemini_calls", "macro_calls", "boardroom_calls", "groq_calls", "groq_429s",
                   "scout_radar_calls", "ranking_calls", "post_mortem_calls",
                   "tape_whisperer_calls", "arbiter_calls", "trap_gate_calls")


@st.cache_resource
def _log_scan_checkpoint() -> dict:
    """Byte offsets + per-day tallies so the AI-usage counters grow incrementally
    instead of re-reading every log file (~200 MB, ~4s) on each refresh."""
    return {"date": "", "files": {}}


def _tally_log_line(line: str, counts: dict) -> None:
    if "[AI-MACRO] Synthesis Complete" in line:
        counts["macro_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[AI-ENGINE] Processed opportunity via Primary Google Gemini" in line:
        counts["boardroom_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[AI-REGIME-ARBITER] Primary Gemini synthesis" in line:
        counts["arbiter_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[BREAKOUT-TRAP-GATE]" in line and "CONFIRMED" in line:
        counts["trap_gate_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[AI-RADAR] AI Scan Complete" in line or "[MTF-CONFLUENCE] Scan Complete" in line:
        counts["scout_radar_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[HEDGING-ORACLE]" in line and "Shield=" in line:
        counts["gemini_calls"] += 1
    elif "[CHIEF-STRATEGIST]" in line and "Asset Hierarchy Updated" in line:
        counts["ranking_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[RELATIVE-STRENGTH] Fleet Alpha Leaders" in line:
        counts["ranking_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[POST-MORTEM-RULE]" in line or "[INSTANT-AUTOPSY]" in line:
        counts["post_mortem_calls"] += 1
        counts["gemini_calls"] += 1
    elif "[TAPE-WHISPERER]" in line:
        counts["tape_whisperer_calls"] += 1
        counts["groq_calls"] += 1
    elif "[GHOST-WALL]" in line or "[RUNNER-ARCHITECT]" in line:
        counts["groq_calls"] += 1
    elif "Processed opportunity via Groq Failover" in line:
        counts["groq_calls"] += 1
    elif "[SRE-ROTATION] Groq 429 Hit" in line:
        counts["groq_429s"] += 1


def _scan_log_file(lp: str, today_str: str) -> dict:
    """Tally only the bytes appended since this file's checkpoint. Returns new counts."""
    state = _log_scan_checkpoint()
    if state["date"] != today_str:
        state["date"] = today_str
        state["files"] = {}
    entry = state["files"].setdefault(lp, {"pos": 0, "counts": dict.fromkeys(_LOG_TALLY_KEYS, 0)})
    try:
        if os.path.getsize(lp) < entry["pos"]:
            entry["pos"] = 0
            entry["counts"] = dict.fromkeys(_LOG_TALLY_KEYS, 0)
    except OSError:
        return {}
    fresh = dict.fromkeys(_LOG_TALLY_KEYS, 0)
    today_b = today_str.encode("utf-8")
    tail = b""
    try:
        with open(lp, "rb") as f:
            f.seek(entry["pos"])
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                segments = (tail + chunk).split(b"\n")
                tail = segments.pop()
                for raw in segments:
                    entry["pos"] += len(raw) + 1
                    if today_b in raw:
                        _tally_log_line(raw.decode("utf-8", "replace"), fresh)
    except OSError:
        return fresh
    for key, val in fresh.items():
        entry["counts"][key] += val
    return fresh


@st.cache_data(ttl=15)
def fetch_ai_usage_stats() -> dict:
    """Computes daily AI credit, call count, and token usage telemetry across Gemini & Groq."""
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    counts = dict.fromkeys(_LOG_TALLY_KEYS, 0)
    for lp in [
        os.path.join(BASE_DIR, "logs", "live_trader.log"),
        os.path.join(BASE_DIR, "logs", "macro_sentinel.log"),
        os.path.join(BASE_DIR, "logs", "main.log"),
        os.path.join(BASE_DIR, "logs", "asset_scout.log"),
        os.path.join(BASE_DIR, "logs", "ofi_daemon.log"),
        os.path.join(BASE_DIR, "logs", "post_mortem_agent.log"),
        os.path.join(BASE_DIR, "logs", "regime_classifier.log"),
        os.path.join(BASE_DIR, "logs", "swing_trader.log"),
        os.path.join(BASE_DIR, "logs", "start_system.log"),
    ]:
        if not os.path.exists(lp):
            continue
        for key, val in _scan_log_file(lp, today_str).items():
            counts[key] += val
    gemini_calls = counts["gemini_calls"]
    macro_calls = counts["macro_calls"]
    boardroom_calls = counts["boardroom_calls"]
    groq_calls = counts["groq_calls"]
    groq_429s = counts["groq_429s"]
    scout_radar_calls = counts["scout_radar_calls"]
    ranking_calls = counts["ranking_calls"]
    post_mortem_calls = counts["post_mortem_calls"]
    tape_whisperer_calls = counts["tape_whisperer_calls"]
    arbiter_calls = counts["arbiter_calls"]
    trap_gate_calls = counts["trap_gate_calls"]

    # Token calculations across all engines
    est_input_tokens = (macro_calls * 500) + (boardroom_calls * 800) + (arbiter_calls * 450) + (trap_gate_calls * 400) + (scout_radar_calls * 650) + (ranking_calls * 450) + (post_mortem_calls * 350) + (tape_whisperer_calls * 250)
    est_output_tokens = (macro_calls * 80) + (boardroom_calls * 100) + (arbiter_calls * 70) + (trap_gate_calls * 60) + (scout_radar_calls * 120) + (ranking_calls * 80) + (post_mortem_calls * 60) + (tape_whisperer_calls * 40)
    total_tokens = est_input_tokens + est_output_tokens

    free_tier_limit = 1500
    free_tier_remaining = max(0, free_tier_limit - gemini_calls)
    est_cost_usd = 0.0 if gemini_calls <= free_tier_limit else ((est_input_tokens * 0.075 / 1e6) + (est_output_tokens * 0.30 / 1e6))

    return {
        "gemini_calls": gemini_calls,
        "macro_calls": macro_calls,
        "boardroom_calls": boardroom_calls,
        "arbiter_calls": arbiter_calls,
        "trap_gate_calls": trap_gate_calls,
        "groq_calls": groq_calls,
        "groq_429s": groq_429s,
        "total_tokens": total_tokens,
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "free_tier_limit": free_tier_limit,
        "free_tier_remaining": free_tier_remaining,
        "est_cost_usd": est_cost_usd,
        "primary_model": getattr(config, "GEMINI_MODEL", "gemini-flash-lite-latest"),
        "failover_model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "groq_pool_size": 3,
        "today_str": today_str
    }


def _missing_dataset_warnings(missing_datasets) -> list[str]:
    """One sentence per arm whose corpus was thinner than it asked for. Absence-tolerant by design.

    ``missing_datasets`` is the quant-validation report's top-level arm -> reasons map
    (``quant_validation.py:768``, also published per-arm at ``:553``), and a reason covers BOTH
    datasets that are absent and datasets that were present but unusable. The validation panel used to
    read that same report and drop the key, so the only human-facing view of a validation run gave no
    way to learn that an arm's numbers rest on fewer datasets than that arm requested — the CLI prints
    them (``:795``) and the panel did not.

    The decision lives here, pure and module-level, so it can be tested without booting this 4,500-line
    script; the panel's ``st.warning`` loop is then the one line this helper cannot prove, and nothing
    else. Never raises: an unreadable report field must not take down the panel around it.
    """
    msgs: list[str] = []
    if not isinstance(missing_datasets, dict):
        return msgs
    for arm, gaps in missing_datasets.items():
        if not gaps or isinstance(gaps, bool):
            continue
        if isinstance(gaps, str):
            raw = [gaps]
        elif isinstance(gaps, (list, tuple, set, frozenset)):
            raw = [str(g) for g in gaps]
        else:
            # A bare number or object names no dataset and no cause. Rendering one would put a reason
            # in front of the reader that the report never actually stated.
            continue
        reasons = [r.strip() for r in raw if r.strip().lower() not in _JUNK_TOKENS]
        if not reasons:
            continue
        msgs.append(
            f"Validation data gap in the {arm} arm — datasets it asked for and did not get: "
            f"{'; '.join(reasons)}. The numbers shown for this arm cover fewer datasets than it "
            "requested, so they are not a measurement of the full corpus its name implies."
        )
    return msgs


def _profit_factor_delta(profit_factor, win_count, loss_count) -> str:
    """Render the fleet's profit factor, or say why there is not one. Absence-tolerant by design.

    The KPI card used to publish `2.62` -- a hardcoded constant -- whenever the closed trades had
    no losing trade to divide by, `1.0` when there were none at all, and `0.0` for the empty
    ledger: a fabricated ratio dressed as a measurement, and a measured zero for a measurement
    that was never taken. The harness axis was fixed the same way (`format_profit_factor` in
    `backtests/backtest_ablation_harness.py`): the ratio is None when it has no denominator, and
    the renderer prints the absence and names its cause.

    `profit_factor` is that ratio, or None when there is no denominator to divide by. The win and
    loss counts are the population it would have been measured over, and they tell the causes
    apart:
      - no closed trades at all -> nothing was measured;
      - closed trades and not one of them a loser -> the ratio has no denominator;
      - closed trades whose every non-win was a breakeven -> same, and the counts say so.

    The decision lives here, pure and module-level, so it can be tested without booting this
    4,500-line script; the panel's ``st.metric`` call is then the one line this helper cannot
    prove, and nothing else. Never raises: an unreadable trade population must not take down the
    card around it.
    """
    try:
        win_count = int(win_count)
        loss_count = int(loss_count)
    except (TypeError, ValueError):
        win_count = loss_count = 0

    ratio = None
    if profit_factor is not None:
        try:
            ratio = float(profit_factor)
        except (TypeError, ValueError):
            ratio = None

    if ratio is not None:
        return f"{ratio:.2f}x Profit Factor ({win_count}W / {loss_count}L)"
    if win_count == 0 and loss_count == 0:
        return "Profit Factor not measured (no closed trades)"
    if loss_count == 0:
        return "Profit Factor not measured (no losing trade, so the ratio has no denominator)"
    return ("Profit Factor not measured (every losing trade was breakeven, so the ratio has no "
            "denominator)")


def _win_rate_value(win_rate_pct, closed_trade_count) -> str:
    """Render the fleet's win rate, or say why there is not one. Absence-tolerant by design.

    The KPI card's win-rate VALUE used to read `0.0% Win Rate` whenever the ledger held no closed
    trade: a measured zero for a measurement that was never taken -- the same mistake the card's
    profit-factor delta was making one cell away, and the same one this branch removed from the
    harness and the backtesters. A win rate is a ratio over closed trades; with none closed there
    is no population to divide by, so there is no rate to publish.

    `win_rate_pct` is the measured rate, or None when it was never measured. `closed_trade_count`
    is the population it would have been measured over, and it is what tells the two apart: a
    fleet that closed trades and won none of them has a real 0.0% win rate, while a fleet that
    closed none has no rate at all.

    Pure and module-level, like `_profit_factor_delta` above, so it can be tested without
    booting this 4,500-line script. Never raises: an unreadable trade population must not take
    down the card around it.
    """
    try:
        closed_trade_count = int(closed_trade_count)
    except (TypeError, ValueError):
        closed_trade_count = 0

    rate = None
    if win_rate_pct is not None:
        try:
            rate = float(win_rate_pct)
        except (TypeError, ValueError):
            rate = None

    if rate is None:
        if closed_trade_count <= 0:
            return "Win Rate not measured (no closed trades)"
        return "Win Rate not measured"
    return f"{rate:.1f}% Win Rate"


def _calibration_panel_values(c_metrics) -> dict:
    """Render the calibration panel's three metrics, or say that none of them was measured.

    The panel used to publish `0%`, `0 Trades`, `0.0000` and a `GOOD` grade whenever the tracker
    had measured nothing -- and a panel with no data at all graded itself GOOD. The `.get(key, 0)`
    defaults were the defect: a key the tracker never produced and a measured zero looked the same
    on screen, so a reader could not tell "the model was wrong every time" from "nothing was
    measured yet".

    `compute_calibration_metrics()` returns three different shapes
    (`model_calibration_tracker.py:112-160`) and they mean three different things:
      - `{}` when the calibration log could not be read: every value is absent, so every cell is
        an absence;
      - `{"total_samples": 0, "brier_score": 0.0, "bins": []}` when the log holds no rows: the
        sample count is a real zero (nothing was evaluated), the win rate and the grade are absent
        keys, and the Brier score is a stand-in for a score with no samples to average over;
      - the full dict once there are samples: a measured count, a measured rate, a measured score
        and the grade that score earned.

    So the count of zero is published as the measurement it is, and the rate, the score and the
    grade are published as absences that name their cause.

    Pure and module-level, like `_win_rate_value` above, so it can be tested without booting this
    4,500-line script. Never raises: an unreadable metrics dict must not take down the panel.
    """
    metrics = c_metrics if isinstance(c_metrics, dict) else {}

    def _number(key):
        raw = metrics.get(key)
        if raw is None or isinstance(raw, bool):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _text(key):
        raw = metrics.get(key)
        if raw is None or isinstance(raw, bool):
            return None
        text = str(raw).strip()
        return text or None

    samples = _number("total_samples")
    if samples is None:
        signals_text = "n/a (calibration metrics unavailable)"
        absent_reason = "calibration metrics unavailable"
    elif samples <= 0:
        # A count of zero is a measurement of zero: nothing was evaluated, and that is the fact.
        signals_text = "0 Trades"
        absent_reason = "no signals evaluated yet"
    else:
        signals_text = f"{int(samples)} Trades"
        absent_reason = "not measured"

    # A rate, a score and a grade need a population to be measured over. With no samples there is
    # none, so they are absences -- never a zero and never a grade nobody earned.
    measured = samples is not None and samples > 0

    rate = _number("overall_win_rate")
    win_rate_text = f"{rate:.1f}%" if (measured and rate is not None) else f"n/a ({absent_reason})"

    score = _number("brier_score")
    brier_text = f"{score:.4f}" if (measured and score is not None) else f"n/a ({absent_reason})"

    grade = _text("calibration_grade")
    grade_text = grade if (measured and grade is not None) else f"not measured ({absent_reason})"

    return {
        "signals_evaluated": signals_text,
        "historical_win_rate": win_rate_text,
        "brier_score": brier_text,
        "calibration_grade": grade_text,
    }


def _optional_float(value):
    """Return the reading as a float, or None when the source carried none.

    A gauge that was never measured has no number to publish, and None is what tells that
    absence apart from a measured zero. Placeholder text ("--", "syncing..."), NULL, NaN
    and unparseable strings are the absence, not a value.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("$", "")
        if not text or text.lower() in ("none", "null", "nan", "n/a", "na", "--", "syncing..."):
            return None
        value = text
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _regime_readings(regime_row, oracle_row) -> dict:
    """Return the regime/oracle gauge readings, or None for each one never measured.

    Every value is a float when the source carried a usable reading and None when it did
    not: no row at all, a missing column, a NULL, a NaN, or unparseable junk. The gauges
    used to fall back on stand-in numbers (ADX 28.5, Hurst 0.62, volatility 0.55, trend
    strength 0.40), so a fleet whose regime daemon had published nothing yet rendered a
    confident-looking regime deck. A gauge with no reading must say the reading was never
    taken.

    `regime_row` is the latest `market_regime` row and `oracle_row` the latest
    `oracle_state` row, or None when the table was empty. Both may be a pandas Series or
    a dict. Pure and module-level, so it can be tested without booting either app, and it
    never raises.
    """
    def _pick(row, key):
        if row is None:
            return None
        try:
            if key not in row:
                return None
        except TypeError:
            return None
        try:
            return _optional_float(row.get(key))
        except AttributeError:
            return None

    return {
        "adx": _pick(regime_row, "adx"),
        "hurst": _pick(regime_row, "hurst"),
        "plus_di": _pick(regime_row, "plus_di"),
        "minus_di": _pick(regime_row, "minus_di"),
        "volatility": _pick(oracle_row, "volatility"),
        "trend_strength": _pick(oracle_row, "trend_strength"),
    }


def _macro_regime(raw) -> str:
    """Normalise the macro regime into a stance the HUD is allowed to show.

    The fleet publishes BULLISH, BEARISH, NEUTRAL and UNKNOWN for `btc_macro_trend`
    (live_oracle.py:404-440), so UNKNOWN is a real published state, not only an absence.
    The HUD used to default the key to "BULLISH", which invented a direction out of a
    reading that had never been taken and announced "MACRO BULLISH (LONG ONLY)". Absent,
    UNKNOWN and junk all mean the same thing here: no stance.

    Returns "BULLISH", "BEARISH", "NEUTRAL" or "UNKNOWN".
    """
    text = str(raw or "").strip().upper()
    if text in ("BULLISH", "BEARISH", "NEUTRAL"):
        return text
    return "UNKNOWN"


def _gauge_display(value, formatter, classify, absent_reason) -> dict:
    """Return what one gauge should show: its value, its delta, and how to dress it.

    `value` is the measured reading or None. When it is None the gauge says the reading
    was never taken, and its delta carries the reason with no arrow and no colour, so an
    absence can never be mistaken for a reading that moved. When the reading exists,
    `formatter` renders it and `classify` turns it into the delta text the panel shows.

    Pure and module-level, so it can be tested without booting either app, and it never
    raises.
    """
    if value is None:
        return {"value": "not measured", "delta": absent_reason, "arrow": "off", "color": "off"}
    try:
        rendered = formatter(value)
    except Exception:
        rendered = "not measured"
    try:
        classified = classify(value)
    except Exception:
        classified = "measured"
    return {"value": rendered, "delta": classified, "arrow": "auto", "color": "normal"}


def _published_number(value, absent_reason: str) -> dict:
    """Return {"value", "measured", "reason"} for a numeric reading.

    A number that no source produced has nothing to publish, and None is what tells that
    absence apart from a measured zero. `float(value or 0.0)` cannot: it invents a zero for a
    missing key, and it overwrites a measured zero with the stand-in it was standing in for.
    """
    number = _optional_float(value)
    if number is None:
        return {"value": None, "measured": False, "reason": absent_reason}
    return {"value": number, "measured": True, "reason": ""}


def _published_boolean(value, absent_reason: str) -> dict:
    """Return {"value", "measured", "reason"} for a tri-state boolean reading.

    A JSON `false` is a measurement: the daemon computed the flag and the answer was no.
    Only a missing or null key is an absence. `bool(None)` is False, so the plain
    `bool(raw)` the code used turned "never measured" into "measured false" and published a
    green badge for it.
    """
    if isinstance(value, bool):
        return {"value": value, "measured": True, "reason": ""}
    if isinstance(value, int) and value in (0, 1):
        return {"value": bool(value), "measured": True, "reason": ""}
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return {"value": True, "measured": True, "reason": ""}
        if text in ("false", "0", "no", "off"):
            return {"value": False, "measured": True, "reason": ""}
    return {"value": None, "measured": False, "reason": absent_reason}


def _recorded_amount(*raw_values) -> float:
    """Return the first usable amount among `raw_values`, or None when none was recorded.

    A deposit tranche is an operator figure: it lives in system_state once the operator
    has recorded it, and in the environment when it was only configured. The HUD used to
    fall back on hardcoded amounts that happened to match one operator's history, so a
    fleet with no recorded deposit still rendered a deposit schedule. None means: not
    recorded.
    """
    for raw in raw_values:
        number = _optional_float(raw)
        if number is not None:
            return number
    return None


def _capital_display(total_deposited, deposit_t1, deposit_t2, deposit_t3) -> dict:
    """Render the HUD's capital figures as the strings a reader sees.

    Each argument is a float when the figure was recorded and None when it was not.
    A figure nobody recorded reads "not recorded" -- never a number borrowed from a
    hardcoded deposit schedule, and never a zero. A recorded zero is a fact and stays a
    number.

    Keys:
      deposited      -- "$250.00", or "not recorded"
      deposited_usdt -- "$250.00 USDT", or "not recorded"
      seed           -- "$55", or "not recorded"
      topup1         -- "$82", or "not recorded"
      topup2         -- "$113", or "not recorded"
      topup2_shown   -- True when the third tranche is a recorded non-zero amount, False
                        when it is a recorded zero, and None when it was never recorded
      tranches       -- "$55 seed + $82 top-up 1 + $113 top-up 2", with "not recorded"
                        standing in for any figure that was never recorded
      tranches_plain -- the same three figures without their labels
    """
    def _amount(value, decimals=2):
        return f"${value:,.{decimals}f}" if value is not None else "not recorded"

    seed = _amount(deposit_t1, 0)
    topup1 = _amount(deposit_t2, 0)
    topup2 = _amount(deposit_t3, 0)
    deposited = _amount(total_deposited)

    labelled = f"{seed} seed + {topup1} top-up 1"
    plain = f"{seed} + {topup1}"
    if deposit_t3 is None:
        labelled += " + not recorded top-up 2"
        plain += " + not recorded"
    elif deposit_t3 > 0:
        labelled += f" + {topup2} top-up 2"
        plain += f" + {topup2}"

    return {
        "deposited": deposited,
        "deposited_usdt": f"{deposited} USDT" if total_deposited is not None else "not recorded",
        "seed": seed,
        "topup1": topup1,
        "topup2": topup2,
        "topup2_shown": None if deposit_t3 is None else (deposit_t3 > 0),
        "tranches": labelled,
        "tranches_plain": plain,
    }


def _profit_display(total_profit, total_roi_pct) -> str:
    """Render the net profit, or say why there is none.

    `total_profit` is the measured profit in USDT and `total_roi_pct` the measured return
    on the recorded capital; both are None when the profit could not be measured. A
    profit needs both a live balance and a recorded capital. With either missing there
    is no profit to publish, and publishing $0.00 would say the fleet broke even.
    """
    if total_profit is None or total_roi_pct is None:
        return "not measured (no recorded capital or no balance telemetry)"
    sign = "+" if total_profit >= 0 else "-"
    return f"{sign}${abs(total_profit):,.2f} USDT ({total_roi_pct:+.2f}%)"


def _published_state(value, absent_reason: str) -> dict:
    """Return what the HUD should publish for one state-backed reading.

    `value` is the raw value the fleet published for a `system_state` key, or the empty
    string when the key was never published. An empty value, a NULL or placeholder text
    means the reading was never taken, so the HUD says so and names the reason instead of
    publishing a stand-in. A value that IS present is returned exactly as the publisher
    wrote it — a recorded zero is a fact and stays a reading.

    Returns {"text": str, "measured": bool, "reason": str}.
    """
    text = str(value if value is not None else "").strip()
    if not text or text.lower() in ("none", "null", "nan", "n/a", "na", "--"):
        return {"text": "not measured", "measured": False, "reason": absent_reason}
    return {"text": text, "measured": True, "reason": ""}


def _published_flag(value, absent_reason: str) -> dict:
    """Return the published flag state, or an absence when the key was never published.

    A control flag and a reading are read the same way but mean different things when they
    are absent: a halt flag that was never published is NOT "not halted". The HUD used to
    default `RISK_HALT` to "0", so a fleet whose risk engine had published nothing showed a
    green "SYSTEM ACTIVE" badge beside the trading controls.

    A published value the HUD does not recognise is not evidence of "off" either. The pre-fix
    code read the halt flag as `!= "0"`, so any spelling other than "0" paused entries; reading
    an unknown spelling as "off" would move that control from fail-safe to fail-open. Only an
    explicit off is read as off, and anything else is read as the control being engaged.

    Returns {"flag": bool, "measured": bool, "reason": str}.
    """
    state = _published_state(value, absent_reason)
    if not state["measured"]:
        return {"flag": False, "measured": False, "reason": state["reason"]}
    text = state["text"].strip().lower()
    return {"flag": text not in ("0", "false", "no", "off"), "measured": True, "reason": ""}


def _stance_color(label: str, measured: bool, bullish: tuple, bearish: tuple) -> str:
    """Colour a published stance by what it says; an absent stance is neutral grey.

    A stance the HUD could not measure must never wear the colour of a measured one: the
    veto pulse defaulted the market regime to "TRENDING" and painted the whole panel green
    off a reading nobody took, and the warden bias was hardcoded green whatever the warden
    said.

    Returns a CSS colour string.
    """
    if not measured:
        return "#94a3b8"
    text = str(label or "").strip().upper()
    if text in bullish:
        return "#00ff66"
    if text in bearish:
        return "#ff003c"
    return "#f59e0b"


def _absence_note(reason: str) -> str:
    """The dimmed parenthetical that says why a reading is missing."""
    return f" <span style='color: #94a3b8; font-size: 0.8rem;'>({html.escape(str(reason))})</span>"


# ── BTC cash-and-carry basis engine: published readings only, never a stand-in book ──────
# The panel used to default every economics field to a plausible figure — a 1000-dollar
# allocation, an 8 percent hurdle, a 100 percent liquidation buffer, a "crossed" margin mode — and
# to a status the engine never claimed. The live telemetry carries five keys for a DISABLED engine,
# so the HUD was publishing a delta-neutral carry book with a 100% liquidation buffer for a
# position that does not exist. Each field is now read on its own, and an absent field says so.
# These four economics thresholds are the panel's OWN — they are not measurements and are labelled
# where they are shown: `panel scale:` on the fee bar, `panel min-buffer threshold:` on the
# liquidation guard, the `±$… panel threshold` on the net-delta posture, and the green rung
# `_BASIS_SAFE_BUFFER_PCT` that guard has to clear before it prints an all-clear.
_BASIS_FEE_SCALE_PCT = 125.0
_BASIS_MIN_BUFFER_PCT = 20.0
_BASIS_NEUTRAL_TOL_USD = 1.0
# The engine publishes its own minimum liquidation buffer as `min_liq_distance_pct`
# (basis_trader.py:836) and that figure drives the amber rung. Green is the panel's own: sitting at
# someone's minimum is not the same as being clear of it.
_BASIS_SAFE_BUFFER_PCT = 30.0
# Telemetry older than this is reported as stale rather than being allowed to look current.
_BASIS_STALE_DAYS = 1.0

# The states the basis engine publishes, with the badge each one earns. A state that is not in
# here was measured but is not a state this panel understands, so it is shown as published rather
# than quietly renamed into a status the engine never claimed.
_BASIS_STATE_BADGES = {
    "ACTIVE_DELTA_NEUTRAL": ("#00ff66", "🟢 ACTIVE 1:1 DELTA-NEUTRAL TWIN"),
    "PENDING_PERP_FILL": ("#ffcc00", "⏳ PENDING PERP MAKER FILL (15s GATE)"),
    # IDLE is the engine's own resting state (`self.state = "IDLE"` at basis_trader.py:155, :161,
    # :410, :477, :503, :691) — running, evaluating the gate, holding no twin position. It is not
    # an all-clear, so it wears the parked slate rather than the active green.
    "IDLE": ("#64748b", "▶️ IDLE — ENGINE RUNNING, NO TWIN POSITION OPEN"),
    "UNWINDING": ("#ff003c", "⚠️ UNWINDING TWIN POSITION"),
    "DISABLED": ("#64748b", "⏸️ PARKED / DISABLED"),
}


# What the coverage headline does NOT count: `age_days` and `stale` are derived from the payload's
# own timestamp rather than published beside it, and `timestamp` is reported on the age line, so
# none of the three is an economics field. The roster itself is derived from `_basis_readings`, so a
# field added there is counted the day it lands instead of waiting for a hand-kept tuple.
_BASIS_COVERAGE_EXCLUDED = ("age_days", "stale", "timestamp")


def _basis_readings(basis_data, now=None) -> dict:
    """Return the cash-and-carry panel's figures, each one a measurement or an absence.

    `basis_data` is the parsed `basis_trader_telemetry` JSON, and may be any subset of the keys —
    today's live payload carries five (`status`, `state`, `reason`, `net_delta`, `timestamp`) for
    an engine that is parked. Every economics field the panel shows is published through
    `_published_number` / `_published_state` on its own, so absence is per field: a missing key
    yields {"value": None, "measured": False, "reason": ...} and never the stand-in the panel used
    to substitute. A recorded zero is a fact and stays a reading, which matters because
    `net_delta = 0.0` IS published today and must keep rendering as a measured `+0.00`.

    `age_days` is derived from the payload's own `timestamp` — a measurement — and is None when no
    usable timestamp was published. `now` is injectable so the age is testable without a clock.

    Pure and module-level: it boots no Streamlit and touches no database, so it can be compiled out
    of the source and called with the live payload shape.
    """
    data = basis_data if isinstance(basis_data, dict) else {}

    def _number(key):
        return _published_number(data.get(key), f"no {key} in basis trader telemetry")

    readings = {
        "state": _published_state(data.get("state"), "no state in basis trader telemetry"),
        "reason": _published_state(data.get("reason"), "no reason in basis trader telemetry"),
        "margin_mode": _published_state(data.get("margin_mode"),
                                        "no margin_mode in basis trader telemetry"),
        "timestamp": _published_state(data.get("timestamp"),
                                      "no timestamp in basis trader telemetry"),
        "funding_rate_8h": _number("funding_rate_8h"),
        "annualized_yield_pct": _number("annualized_yield_pct"),
        "hurdle_rate_pct": _number("hurdle_rate_pct"),
        "unwind_hurdle_pct": _number("unwind_hurdle_pct"),
        "capital_allocated_usd": _number("capital_allocated_usd"),
        "cumulative_funding_harvested": _number("cumulative_funding_harvested"),
        # Two writers, two key names. The active engine publishes `net_delta_usd`
        # (basis_trader.py:823); the DISABLED parking path publishes `net_delta`
        # (basis_trader.py:920). Prefer the engine's own computed key and fall back to the parked
        # placeholder. Membership decides, never truthiness: `or` and `.get(k, default)` silently
        # drop a measured 0.0, which is the defect class this panel was built to end.
        "net_delta": _published_number(
            data["net_delta_usd"] if "net_delta_usd" in data else data.get("net_delta"),
            "no net delta in basis trader telemetry"),
        "holding_days": _number("holding_days"),
        "min_holding_days": _number("min_holding_days"),
        "fee_hurdle_target_usd": _number("fee_hurdle_target_usd"),
        "fee_amortized_pct": _number("fee_amortized_pct"),
        "liquidation_price": _number("liquidation_price"),
        "distance_to_liq_pct": _number("distance_to_liq_pct"),
        # The engine's own minimum liquidation buffer, published next to the distance above it
        # (basis_trader.py:836). A published 0.0 is a threshold the engine actually enforces, so it
        # is read through the same path as every other figure rather than defaulted.
        "min_liq_distance_pct": _number("min_liq_distance_pct"),
        "basis_spread_usd": _number("basis_spread_usd"),
        "basis_dislocation_pct": _number("basis_dislocation_pct"),
    }

    age_days = None
    if readings["timestamp"]["measured"]:
        raw_ts = str(readings["timestamp"]["text"]).strip().replace("Z", "+00:00")
        try:
            published_at = datetime.fromisoformat(raw_ts)
        except (TypeError, ValueError):
            published_at = None
        if published_at is not None:
            if published_at.tzinfo is None:
                published_at = pytz.utc.localize(published_at)
            reference = now if isinstance(now, datetime) else datetime.now(pytz.utc)
            if reference.tzinfo is None:
                reference = pytz.utc.localize(reference)
            age_days = (reference - published_at).total_seconds() / 86400.0
    readings["age_days"] = age_days
    readings["stale"] = bool(age_days is not None and age_days >= _BASIS_STALE_DAYS)
    return readings


# ── Bayesian Markov regime deck: published readings only, never a model's prior ───────────
# The deck used to read `markov_regime_telemetry` through `query_system_state`, which hands back the
# value only and so cannot see the row's `updated_at`. A row `regime_classifier.py:1218-1221` wrote
# ~35 hours earlier therefore printed under a heading claiming a live forward filter, while every
# automated consumer of the same row -- `super_bot.py:718-736`, `super_bot.py:909-924`,
# `inflight_shepherd.py:594-615`, `swing_trader.py:928-944` -- refuses it beyond
# `config.REGIME_MAX_AGE_SEC`. It also instantiated `MarkovRegimeEngine` to fill a missing row, and
# because that constructor is pure (`markov_regime.py:33-49`: uniform `pi`, the hardcoded prior `P`,
# no dataset, no DB) every figure out of it was the model's assumption about the world rather than a
# measurement of this one. Both defects end here: the age is measured from the row's own stamp and
# absence is published as absence.
_REGIME_MAX_AGE_FALLBACK_SEC = 900.0
# The same default the consumers fall back to, not a second threshold number: a HUD with its own
# softer bound would keep advertising a row the fleet has already stopped trusting.

# The deck's OWN cascade-risk rungs -- the producer publishes a probability, not a verdict -- so the
# card prints each number beside the label it earned.
_MARKOV_BD_ALARM_PCT = 0.50
_MARKOV_BD_WARN_PCT = 0.30
_MARKOV_BD_HIGH_PCT = 0.60
_MARKOV_BD_ELEV_PCT = 0.35
# The entry hysteresis gate the horizon bars are drawn against, and the fee shield the engine's own
# tradability gate states in `markov_regime.py:329-341`. Named here so the panel never prints a bare
# threshold.
_MARKOV_HYSTERESIS_GATE = 0.65
_MARKOV_FEE_SHIELD_BARS = 3.5


def _regime_max_age_sec() -> float:
    """Return the fleet's own regime-age bound, or the consumers' 900.0 when config cannot answer.

    `config.REGIME_MAX_AGE_SEC` (config.py:109) is the single threshold every automated consumer
    enforces, so the HUD reads it rather than carrying its own number. A missing, zero, negative or
    non-numeric value falls back to 900.0 -- the consumers' own default -- and not to 0.0, which
    would call every row stale, or to a silent pass that would call every row fresh.
    """
    try:
        bound = _optional_float(getattr(config, "REGIME_MAX_AGE_SEC", None))
    except Exception:
        return _REGIME_MAX_AGE_FALLBACK_SEC
    if bound is None or bound <= 0.0:
        return _REGIME_MAX_AGE_FALLBACK_SEC
    return float(bound)


def _markov_regime_readings(raw_value, published_at, now=None, max_age_sec=None) -> dict:
    """Return the Markov deck's figures, each one a measurement or an absence.

    `raw_value` is the `markov_regime_telemetry` row's JSON text (or an already-parsed dict), and
    `published_at` is that row's own `updated_at`: `regime_classifier.py:1218-1221` writes the row
    with `datetime.now(UTC).isoformat()` (`:1136`), so the stamp IS the publish instant and the
    payload carries no timestamp of its own.

    The contract is `markov_regime.py:387 process_bars()`, which publishes exactly ten keys on BOTH
    its paths (`:398-413` short frame, `:445-460` filtered): `dominant_regime`,
    `state_probabilities`, `expected_durations_bars`, `breakdown_risk_4step`, `forecast_4step`,
    `transition_matrix`, `stationary_distribution`, `is_tradable`, `tradability_reason`,
    `sizing_multiplier`. Two shapes matter. `stationary_distribution` is a LIST of three rounded
    floats (`:409`, `:456`) -- a dict-shaped value is the shape the deleted fallback invented, so it
    is an absence here rather than a guess about which slot meant what. And the forecast is published
    as `forecast_4step` (`:407`, `:454`); the name the deck used to read first is only ever returned
    inside `forecast_k_steps` (`:360`) and never leaves it into the payload.

    Every figure goes through `_published_number` / `_published_boolean` / `_published_state` on its
    own, so absence is per field and a missing key never becomes a stand-in. A recorded zero is a
    fact and stays a reading -- the live row carries `BEAR_CASCADE: 0.0002`, and `float(raw or 0.0)`
    erases the difference between that and a key nobody wrote. `sizing_multiplier` runs the other
    way: the producer clamps it into `[0.20, 1.0]` (`markov_regime.py:317-327`), so a measured zero
    is not a reading it can publish, and an absence must never be defaulted to 1.0 -- the largest
    figure it can emit, and the riskiest possible stand-in.

    `age_sec` is measured from the row's own stamp and is None when no usable stamp came with it --
    including when the stamp is ahead of this clock, which is refused rather than clamped, so
    `stamp_is_future` and `age_absence_reason` say which failure happened. `stale` compares that age
    against `config.REGIME_MAX_AGE_SEC`. `now` and `max_age_sec` are
    injectable so the age path is testable without a clock. Pure and module-level: it boots no
    Streamlit, opens no database, and imports no model.
    """
    # The engine's own regime roster (`markov_regime.py:29-31`), with the short field name the deck
    # addresses each reading by. Named here rather than spelled out at each call site so the deck's
    # labels, this reader's fields and the producer's key set have one thing to disagree with.
    _MARKOV_STATE_KEYS = (
        ("chop", "CHOP"),
        ("bull", "BULL_EXPANSION"),
        ("bear", "BEAR_CASCADE"),
    )

    data = raw_value if isinstance(raw_value, dict) else {}
    # Finding #8: `bool(data)` alone called three different situations by one name. Whether the row
    # carries anything at all, and whether what it carries IS a telemetry object, are separate
    # measurements -- and they blame different culprits (the scheduler vs the producer).
    row_written = (isinstance(raw_value, dict)
                   or bool(isinstance(raw_value, str) and raw_value.strip()))
    row_is_object = isinstance(raw_value, dict)
    if isinstance(raw_value, str) and raw_value.strip():
        try:
            parsed = json.loads(raw_value)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            data = parsed
            row_is_object = True

    def _from_payload(payload_key):
        """The value published under `payload_key`, or None when the engine wrote no such key.

        Membership decides, never truthiness and never a two-argument `.get()`: `or` and
        `.get(key, default)` both overwrite a measured zero with the stand-in that was standing in
        for it, which is the defect class this reader exists to end.
        """
        if payload_key not in data:
            return None
        return data[payload_key]

    def _state_map(payload_key, prefix):
        """One reading per regime state, out of a dict the producer keyed by state name."""
        raw = _from_payload(payload_key)
        raw = raw if isinstance(raw, dict) else {}
        return {
            f"{prefix}_{short}": _published_number(
                raw[state] if state in raw else None, f"no {state} in {payload_key}")
            for short, state in _MARKOV_STATE_KEYS
        }

    fields = {
        "dominant_regime": _published_state(
            _from_payload("dominant_regime"), "no dominant_regime in markov regime telemetry"),
        "breakdown_risk": _published_number(
            _from_payload("breakdown_risk_4step"),
            "no breakdown_risk_4step in markov regime telemetry"),
        # Tri-state, not a truthiness test: `bool(m_tel.get("is_tradable", True))` turned a missing
        # key into the green badge, and a published `false` into the same reading as no reading.
        "is_tradable": _published_boolean(
            _from_payload("is_tradable"), "no is_tradable in markov regime telemetry"),
        # The reason is the engine's own alibi, so an absent one is missed, never supplied by the
        # panel: `m_tel.get("tradability_reason", "Tradable")` put words in the engine's mouth.
        "tradability_reason": _published_state(
            _from_payload("tradability_reason"),
            "no tradability_reason in markov regime telemetry"),
        "sizing_multiplier": _published_number(
            _from_payload("sizing_multiplier"),
            "no sizing_multiplier in markov regime telemetry"),
    }
    fields.update(_state_map("state_probabilities", "prob"))
    fields.update(_state_map("expected_durations_bars", "dur"))
    fields.update(_state_map("forecast_4step", "fc"))

    # The equilibrium is the published 3-LIST (`markov_regime.py:409`, `:456`) and nothing else.
    # A short or long list, a dict or a string names no slot, so all three readings are absences
    # rather than a guess about which entry the producer meant.
    raw_stat = _from_payload("stationary_distribution")
    stat_slots = raw_stat if isinstance(raw_stat, (list, tuple)) and len(
        raw_stat) == len(_MARKOV_STATE_KEYS) else None
    for index, (short, state) in enumerate(_MARKOV_STATE_KEYS):
        fields[f"stat_{short}"] = (
            _published_number(stat_slots[index], f"no {state} in stationary_distribution")
            if stat_slots is not None else
            {"value": None, "measured": False,
             "reason": f"no {state} in stationary_distribution"})
    stat_values = None
    if stat_slots is not None:
        candidate = [_optional_float(one) for one in stat_slots]
        if all(one is not None for one in candidate):
            stat_values = candidate
    fields["equilibrium"] = (
        {"value": stat_values, "rows": tuple(stat_values), "measured": True, "reason": ""}
        if stat_values is not None else
        {"value": None, "rows": (), "measured": False,
         "reason": ("no equilibrium vector: stationary_distribution was not published as a "
                    "3-vector")})

    # Two derived figures. Each is a reading only when every figure it is computed from is one:
    # a maximum over a partial probability set, or a duration looked up under a regime that was
    # never named, is a guess about the missing entries rather than a measurement.
    probs_measured = all(fields[f"prob_{short}"]["measured"] for short, _ in _MARKOV_STATE_KEYS)
    fields["confidence"] = (
        {"value": max(fields[f"prob_{short}"]["value"] for short, _ in _MARKOV_STATE_KEYS),
         "measured": True, "reason": ""}
        if probs_measured else
        {"value": None, "measured": False,
         "reason": "no confidence: every state probability has to be measured first"})
    short_by_state = {state: short for short, state in _MARKOV_STATE_KEYS}
    dom_st = fields["dominant_regime"]
    # `.get`, never a subscript. A row whose `dominant_regime` parses as a measured state string yet
    # names none of the engine's three states -- "RANGE", 7, ["CHOP"], the shapes `json.loads` hands
    # back -- has no duration to look up, and it must publish that as an absence rather than raise
    # `KeyError` and take the whole `tab_risk` block down with it: brief constraint 9 counts a
    # crashed panel as worse than an honest absence, and the crash would leave no `FEED_FAILURES`
    # entry either. The absence reason names WHICH of the three causes applies, so the operator is
    # pointed at the producer (a state the engine never defines) and not at the scheduler.
    dom_short = short_by_state.get(str(dom_st["text"]).strip()) if dom_st["measured"] else None
    dom_dur_st = fields.get(f"dur_{dom_short}") if dom_short is not None else None
    if dom_dur_st is not None and dom_dur_st["measured"]:
        fields["dominant_duration"] = {"value": dom_dur_st["value"], "measured": True, "reason": ""}
    elif dom_st["measured"] and dom_short is None:
        fields["dominant_duration"] = (
            {"value": None, "measured": False,
             "reason": ("no dominant duration: dominant_regime names none of the engine's states "
                        "(" + " / ".join(one_state for _one_short, one_state
                                         in _MARKOV_STATE_KEYS) + ")")})
    else:
        fields["dominant_duration"] = (
            {"value": None, "measured": False,
             "reason": ("no dominant duration: the dominant regime or its expected duration "
                        "was not measured")})

    # The transition matrix, or no matrix at all. There is no stand-in: `default_P` was a
    # byte-for-byte copy of the engine's prior (`markov_regime.py:39-43`) which would silently
    # diverge the day that prior changed, and `process_bars` puts A matrix in the payload on both
    # paths (`:408`, `:455`), so the only row without one is a row that does not exist. Which
    # matrix is not knowable from the row: the short-frame path hands back `self.P.tolist()` --
    # that same hardcoded prior -- without ever running `estimate_transition_matrix` (`:416`), so
    # a measured matrix may be estimated from labelled bars or may be the default, and the deck's
    # caption therefore claims publication only.
    raw_matrix = _from_payload("transition_matrix")
    matrix_rows = None
    if isinstance(raw_matrix, (list, tuple)) and len(raw_matrix) == len(_MARKOV_STATE_KEYS):
        candidate_rows = []
        for row in raw_matrix:
            if not isinstance(row, (list, tuple)) or len(row) != len(_MARKOV_STATE_KEYS):
                candidate_rows = None
                break
            numbers = [_optional_float(one) for one in row]
            if any(one is None for one in numbers):
                candidate_rows = None
                break
            candidate_rows.append(numbers)
        matrix_rows = candidate_rows
    fields["transition_matrix"] = (
        {"value": matrix_rows, "rows": tuple(tuple(one) for one in matrix_rows),
         "measured": True, "reason": ""}
        if matrix_rows is not None else
        {"value": None, "rows": (), "measured": False,
         "reason": "no 3x3 transition_matrix in markov regime telemetry"})

    readings = dict(fields)
    readings["field_names"] = tuple(sorted(fields))
    readings["coverage"] = {
        "total": len(readings["field_names"]),
        "unmeasured": sum(1 for name in readings["field_names"] if not fields[name]["measured"]),
    }
    readings["unmeasured_fields"] = tuple(
        name for name in readings["field_names"] if not fields[name]["measured"])

    bound = _REGIME_MAX_AGE_FALLBACK_SEC if max_age_sec is None else _optional_float(max_age_sec)
    if bound is None or bound <= 0.0:
        bound = _REGIME_MAX_AGE_FALLBACK_SEC
    readings["max_age_sec"] = float(bound)

    readings["published_at"] = _published_state(
        published_at, "no updated_at stamp on the markov regime row")
    age_sec = None
    if readings["published_at"]["measured"]:
        # The fleet's own parse shape: tolerate a `Z` suffix and localise a naive stamp to UTC,
        # rather than treating a `CURRENT_TIMESTAMP`-shaped row as if it were written right now.
        raw_ts = str(readings["published_at"]["text"]).strip().replace("Z", "+00:00")
        try:
            stamp = datetime.fromisoformat(raw_ts)
        except (TypeError, ValueError):
            stamp = None
        if stamp is not None:
            if stamp.tzinfo is None:
                stamp = pytz.utc.localize(stamp)
            reference = now if isinstance(now, datetime) else datetime.now(pytz.utc)
            if reference.tzinfo is None:
                reference = pytz.utc.localize(reference)
            age_sec = (reference - stamp).total_seconds()
    # Finding #7 (controller ruling): a stamp AHEAD of the clock is refused, not clamped. The fleet
    # folds the negative away (`inflight_shepherd.py:611 age = max(0.0, ...)`) which is safe for a
    # veto but would here print a fabricated "0s old" freshness claim -- the defect class this panel
    # was rebuilt to end -- so the deck is DELIBERATELY STRICTER than the fleet on future stamps.
    readings["stamp_is_future"] = bool(age_sec is not None and age_sec < 0.0)
    readings["age_sec"] = None if readings["stamp_is_future"] else age_sec
    readings["age_known"] = readings["age_sec"] is not None
    if readings["stamp_is_future"]:
        readings["age_absence_reason"] = ("stamp is in the future — age not measurable")
    elif readings["age_known"]:
        readings["age_absence_reason"] = ""
    elif readings["published_at"]["measured"]:
        readings["age_absence_reason"] = ("stamp is not a parseable instant — age not measurable")
    else:
        readings["age_absence_reason"] = readings["published_at"]["reason"]
    readings["stale"] = bool(readings["age_sec"] is not None and readings["age_sec"] > float(bound))
    readings["row_present"] = bool(data)
    readings["row_written"] = bool(row_written)
    readings["row_is_telemetry_object"] = bool(row_is_object)
    # "Live" needs both a stamp inside the bound the fleet enforces AND a row that measured
    # something. A fresh row carrying nothing is not a live forward filter either.
    readings["heading_is_live"] = bool(readings["age_known"] and not readings["stale"]
                                       and readings["row_present"])

    readings["states"] = tuple(state for _short, state in _MARKOV_STATE_KEYS)
    readings["by_state"] = tuple(
        {"short": short, "state": state, "prob": fields[f"prob_{short}"],
         "dur": fields[f"dur_{short}"], "fc": fields[f"fc_{short}"],
         "stat": fields[f"stat_{short}"]}
        for short, state in _MARKOV_STATE_KEYS)
    return readings


# ── Dark Financial Terminal Aesthetic ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0,1&family=IBM+Plex+Mono:ital,wght@0,400;0,600;0,700;1,400&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --terminal-bg: #0a0a0c;
    --panel-bg: #111114;
    --accent-amber: #d4a017;
    --accent-cyan: #00bcd4;
    --accent-emerald: #00c853;
    --accent-rose: #e91e63;
    --text-primary: #e8e6e1;
    --text-muted: #6b6b6f;
    --grid-line: rgba(212, 160, 23, 0.06);
    --border-subtle: rgba(255, 255, 255, 0.06);
}

/* Global reset */
.stApp {
    background: var(--terminal-bg) !important;
    background-image: 
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px) !important;
    background-size: 40px 40px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Typography hierarchy */
h1, h2, h3, h4, h5, h6 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

p, span, div, label {
    font-family: 'Space Grotesk', sans-serif !important;
}

code, pre, .monospace {
    font-family: 'IBM Plex Mono', 'Fira Code', monospace !important;
    font-size: 0.875em !important;
}

/* Sidebar refinement */
section[data-testid="stSidebar"] {
    background: var(--panel-bg) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--accent-amber) !important;
}

/* Metric cards — terminal-style data blocks */
div[data-testid="stMetric"] {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: 2px solid var(--accent-amber) !important;
    border-radius: 2px !important;
    padding: 16px 20px !important;
    position: relative;
    overflow: hidden;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-amber), transparent);
    opacity: 0.4;
}

[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
}

[data-testid="stMetricLabel"] p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

/* Tab navigation — understated elegance */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    padding: 12px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease;
}

button[data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(255, 255, 255, 0.02) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-amber) !important;
    border-bottom: 2px solid var(--accent-amber) !important;
    background: transparent !important;
}

/* Data tables — clean terminal readout */
div[data-testid="stDataFrame"] {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 2px !important;
    overflow: hidden;
}

table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

thead th {
    background: rgba(212, 160, 23, 0.08) !important;
    color: var(--accent-amber) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.75rem !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

tbody td {
    border-bottom: 1px solid var(--border-subtle) !important;
    color: var(--text-primary) !important;
}

tbody tr:hover {
    background: rgba(255, 255, 255, 0.02) !important;
}

/* Status badges */
.badge-green {
    background: rgba(0, 200, 83, 0.12) !important;
    color: var(--accent-emerald) !important;
    border: 1px solid var(--accent-emerald) !important;
    padding: 3px 10px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

.badge-red {
    background: rgba(233, 30, 99, 0.12) !important;
    color: var(--accent-rose) !important;
    border: 1px solid var(--accent-rose) !important;
    padding: 3px 10px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

.badge-neutral {
    background: rgba(148, 163, 184, 0.12) !important;
    color: #94a3b8 !important;
    border: 1px solid #94a3b8 !important;
    padding: 3px 10px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

.badge-cyan {
    background: rgba(0, 188, 212, 0.12) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important;
    padding: 3px 10px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

.badge-orange {
    background: rgba(212, 160, 23, 0.12) !important;
    color: var(--accent-amber) !important;
    border: 1px solid var(--accent-amber) !important;
    padding: 3px 10px !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
}

/* Position cards */
.pos-card {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 3px solid var(--accent-cyan) !important;
    border-radius: 2px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    position: relative;
}

.pos-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
    opacity: 0.3;
}

/* Buttons and controls */
button[kind="primary"] {
    background: var(--accent-amber) !important;
    color: var(--terminal-bg) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-size: 0.75rem !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease;
}

button[kind="primary"]:hover {
    background: #e6b422 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(212, 160, 23, 0.3);
}

button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 2px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease;
}

button[kind="secondary"]:hover {
    border-color: var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}

/* Expanders and containers */
details {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 2px !important;
    margin-bottom: 8px !important;
}

summary {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
    padding: 12px 16px !important;
}

/* Alert banners */
div[data-testid="stAlert"] {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-left: 3px solid var(--accent-amber) !important;
    border-radius: 2px !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--terminal-bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border-subtle);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-amber);
}

/* Subtle animation for live data */
@keyframes pulse-amber {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.live-indicator {
    animation: pulse-amber 2s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# ── Data Feed Health Banner ──────────────────────────────────────────────────
# Failures recorded by query_db/execute_safe_write during the *previous* pass.
# A monitoring page that silently renders zeros is worse than one that admits it is blind.
_carried_feed_failures = st.session_state.pop("_carried_feed_failures", {})
if _carried_feed_failures:
    _ff_body = "\n".join(
        f"- `{feed}` — {err}" for feed, err in list(_carried_feed_failures.items())[:12]
    )
    _ff_extra = "" if len(_carried_feed_failures) <= 12 else f"\n- …{len(_carried_feed_failures) - 12} more"
    st.warning(
        f"⚠️ **{len(_carried_feed_failures)} data feed(s) failed on the last refresh.** "
        f"Affected sections fall back to empty/placeholder values — treat them as unknown, not as zero.\n\n"
        f"{_ff_body}{_ff_extra}",
        icon="📡"
    )

# ── Sidebar Controls & Configuration ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 20px 0; border-bottom: 1px solid var(--border-subtle); margin-bottom: 20px;">
        <h2 style="font-family: 'DM Serif Display', serif; color: var(--accent-amber); margin: 0; font-size: 1.6rem; letter-spacing: -0.01em;">Sovereign Swarm</h2>
        <p style="font-family: 'Space Grotesk', sans-serif; color: var(--text-muted); margin: 8px 0 0 0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em;">Autonomous Trading Fleet</p>
    </div>
    """, unsafe_allow_html=True)
    
    # P0: Auto-Refresh Toggle Engine
    st.markdown("**Telemetry Refresh**")
    auto_refresh_enabled = st.checkbox("Live Auto-Refresh (4s)", value=True, help="Refreshes HUD telemetry automatically every 4 seconds.")
    if auto_refresh_enabled and AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=4000, key="sovereign_swarm_hud_autorefresh")
    
    st.markdown("---")

    # P1: Operator Safe Controls Drawer (Protected by Password Auth Gate)
    with st.expander("🛠️ Fleet Operator Controls", expanded=False):
        import hashlib
        APP_PASSWORD = os.getenv("APP_PASSWORD")
        if not APP_PASSWORD:
            st.error("🔒 Operator Controls Disabled: APP_PASSWORD is not set in environment.")
        elif "operator_authenticated" not in st.session_state:
            st.session_state.operator_authenticated = False
            
        if APP_PASSWORD and not st.session_state.get("operator_authenticated", False):
            st.markdown("##### 🔐 Operator Authentication")
            password_input = st.text_input("Operator Password", type="password", key="operator_pwd_gate", help="Enter APP_PASSWORD from .env to unlock fleet mutation commands")
            if st.button("Unlock Controls", key="btn_unlock_op_controls", width="stretch"):
                pwd_hash = hashlib.sha256(password_input.encode()).hexdigest()
                expected_hash = hashlib.sha256(APP_PASSWORD.encode()).hexdigest()
                if pwd_hash == expected_hash:
                    st.session_state.operator_authenticated = True
                    st.success("✅ Operator Controls Unlocked")
                    st.rerun()
                else:
                    st.error("❌ Invalid password")
            st.info("🔒 Enter password to unlock pause/resume and exchange re-sync controls.")
        else:
            if st.button("🔒 Lock Controls", key="btn_lock_op_controls", width="stretch"):
                st.session_state.operator_authenticated = False
                st.rerun()
            st.markdown("##### ⚙️ Operational Overrides")
            curr_op_halt = state_flag_now("OPERATOR_HALT")
            curr_auto_halt = state_flag("RISK_HALT")
            pause_entries = st.toggle(
                "⏸️ Pause Entries (Close-Only)",
                value=curr_op_halt,
                help="Sticky manual pause (`OPERATOR_HALT`). Halts all new trade entries across the fleet while letting existing positions trail safely to SL/TP."
            )
            if curr_auto_halt and not curr_op_halt:
                st.caption("⚠️ *Auto Risk Cap Halt is currently active (margin/slot ceiling).*")
            if pause_entries != curr_op_halt:
                new_val = "1" if pause_entries else "0"
                wrote = execute_safe_write(TRADING_DATA_DB, "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('OPERATOR_HALT', ?, datetime('now'))", (new_val,))
                persisted = read_state_now("OPERATOR_HALT")
                if wrote and str(persisted) == new_val:
                    st.toast(f"Fleet State Updated (verified): {'Entries Paused (Close-Only)' if pause_entries else 'Manual Pause Cleared'}")
                    st.rerun()
                else:
                    st.warning(
                        f"⚠️ PAUSE COMMAND DID NOT LAND — write {'succeeded' if wrote else 'FAILED'}, "
                        f"ledger read-back is {persisted!r} instead of {new_val!r}. The fleet is still in its "
                        f"previous entry state; do not rely on this toggle until it re-reads as intended."
                    )

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Force Exchange Re-Sync", width="stretch", help="Dispatches an immediate re-sync signal to the Reconciliation Engine."):
                wrote = execute_safe_write(TRADING_DATA_DB, "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('force_resync', '1', datetime('now'))", ())
                persisted = str(read_state_now("force_resync"))
                if wrote and persisted == "1":
                    st.toast("✅ Force Re-Sync signal dispatched to Reconciliation Engine!")
                elif wrote:
                    st.toast(f"✅ Force Re-Sync committed; the engine has already moved the flag to {persisted!r}")
                else:
                    st.warning("⚠️ Force Re-Sync signal was NOT persisted to trading_data.db — the Reconciliation Engine will not see it.")

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("📧 Send 1h Rollup Email", width="stretch", help="Immediately synthesizes the last 60 minutes of 5-minute briefings and sends the 1-Hour Executive Briefing email via Gmail SMTP."):
                try:
                    import hourly_briefing
                    import importlib
                    importlib.reload(hourly_briefing)
                    with st.spinner("Dispatching 1h Executive Rollup Email..."):
                        sent_ok = hourly_briefing.run_single_hourly_briefing(force=True)
                        if sent_ok:
                            st.toast("🏆 1-Hour Executive Rollup Email dispatched successfully!")
                            st.success("🏆 1-Hour Executive Rollup Email dispatched to your inbox!")
                        else:
                            st.error("Email dispatch returned false. Check SMTP logs.")
                except Exception as em_ui_err:
                    st.error(f"Failed to dispatch email: {em_ui_err}")

    st.markdown("---")
    live_bal_val = num(query_system_state("balance", ""), None)
    # The operator's capital is a recorded figure (system_state) or a configured one (the
    # environment). Neither was recorded on this fleet, so the HUD says so -- it used to
    # publish a hardcoded deposit schedule and a profit computed against it.
    total_deposited = _recorded_amount(
        query_system_state("user_total_deposit", ""),
        os.getenv("INITIAL_BALANCE", ""),
    )
    deposit_t1 = _recorded_amount(
        query_system_state("initial_deposit_tranche_1", ""),
        os.getenv("INITIAL_DEPOSIT_TRANCHE_1", ""),
    )
    deposit_t2 = _recorded_amount(
        query_system_state("topup_deposit_tranche_2", ""),
        os.getenv("TOPUP_DEPOSIT_TRANCHE_2", ""),
    )
    deposit_t3 = _recorded_amount(
        query_system_state("topup_deposit_tranche_3", ""),
        os.getenv("TOPUP_DEPOSIT_TRANCHE_3", ""),
    )
    capital = _capital_display(total_deposited, deposit_t1, deposit_t2, deposit_t3)
    bal_display = f"${live_bal_val:,.2f}" if live_bal_val is not None else "Syncing..."

    st.write(f"**Connected Exchange:** Bitget UTA (Live)")
    if live_bal_val is not None:
        st.write(f"**Current Equity:** `${live_bal_val:.2f} USDT`")
    else:
        st.write(f"**Current Equity:** `⚠️ Telemetry Offline (Syncing...)`")
    if live_bal_val is not None and total_deposited is not None:
        total_profit = live_bal_val - total_deposited
        total_roi_pct = (total_profit / total_deposited) * 100.0 if total_deposited > 0 else 0.0
    else:
        total_profit = None
        total_roi_pct = None
    # One colour for every consumer of the profit, computed once here so no f-string can
    # compare None against 0.
    profit_color = "#00ff66" if (total_profit is not None and total_profit >= 0) else "#ff003c"
    st.write(f"**Deposited Capital:** `{capital['deposited_usdt']}`")
    tranche_str = f"↳ Seed: `{capital['seed']}` | Top-up 1: `{capital['topup1']}`"
    if capital["topup2_shown"] is None:
        tranche_str += " | Top-up 2: `not recorded`"
    elif capital["topup2_shown"]:
        tranche_str += f" | Top-up 2: `{capital['topup2']}`"
    st.caption(tranche_str)
    if total_profit is None:
        st.markdown(f"**Real Net Profit:** <span style='font-weight:700;'>{_profit_display(total_profit, total_roi_pct)}</span>", unsafe_allow_html=True)
    else:
        profit_sign = "+$" if total_profit >= 0 else "-$"
        st.markdown(f"**Real Net Profit:** <span style='color:{profit_color}; font-weight:700;'>{profit_sign}{abs(total_profit):.2f} USDT ({total_roi_pct:+.2f}%)</span>", unsafe_allow_html=True)
    local_ip = get_local_ip()
    st.write(f"**Local Mobile URL:** `http://{local_ip}:8501`")
    st.markdown("---")
    
    # AI Engine & Credit Telemetry in Sidebar
    ai_side = fetch_ai_usage_stats()
    st.markdown("##### 🧠 AI Engine & Free Tier Quota")
    st.write(f"**Primary (Gemini):** `{ai_side['gemini_calls']} calls` | 🟢 Active")
    st.write(f"**Failover (Groq):** `{ai_side['groq_pool_size']} Keys Pool` | 🟢 Standby")
    st.write(f"**Est. Daily Cost:** `${ai_side['est_cost_usd']:.2f} (100% Free)`")
    with st.expander("⚡ 6 Specialized AI Engines", expanded=False):
        st.caption("🟢 **Liquidity Sweep Gate:** Active (Fakeout Shield)")
        st.caption("🟢 **Early Reversal Sentinel:** Active (Trailing Stops)")
        st.caption("🟢 **Dynamic Exit Geometry:** Active (Structure SL/TP)")
        df_rule_cnt = query_db(TRADING_LEDGER_DB, "SELECT COUNT(*) as cnt FROM learned_rules")
        live_rule_count = int(df_rule_cnt.iloc[0]["cnt"]) if not df_rule_cnt.empty else 0
        rule_cnt_str = f"{live_rule_count} Rules" if live_rule_count > 0 else "Active"
        st.caption(f"🟢 **Rule Enforcement Gate:** Active ({rule_cnt_str})")
        st.caption("🟢 **Funding Carry Shield:** Active (Pre-Settlement)")
        st.caption("🟢 **Session Macro Blueprint:** Active (Every 6h / 4x Daily)")
        # Database Quick Check status
        integ_report = {"is_ok": True, "is_stale": False, "is_corrupt": False, "stale_dbs": [], "corrupt_dbs": [], "latest_utc_dt": None}
        try:
            if os.path.exists(TRADING_DATA_DB):
                with connect_readonly(TRADING_DATA_DB) as _conn:
                    integ_report = check_database_integrity(_conn)
        except Exception:
            integ_report["is_ok"] = False
            integ_report["is_stale"] = True

        if integ_report["is_ok"]:
            st.caption("🟢 **Database Integrity:** Verified OK")
        elif integ_report["is_stale"]:
            last_ts_str = integ_report["latest_utc_dt"].strftime('%H:%M:%S UTC') if integ_report.get("latest_utc_dt") else "UNKNOWN"
            st.caption(f"🟠 **Database Integrity:** SIGNAL STALE (last verified {last_ts_str}, daemon not publishing)")
        else:
            st.caption("🚨 **Database Integrity:** CRITICAL ERROR")
    st.markdown("---")
    st.caption(f"Last HUD Refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

# ── Top Database Integrity Alert Banner ─────────────────────────────────────────
top_integ_report = {"is_ok": True, "is_stale": False, "is_corrupt": False, "stale_dbs": [], "corrupt_dbs": []}
try:
    if os.path.exists(TRADING_DATA_DB):
        with connect_readonly(TRADING_DATA_DB) as _conn:
            top_integ_report = check_database_integrity(_conn)
except Exception:
    top_integ_report["is_ok"] = False
    top_integ_report["is_stale"] = True

if not top_integ_report["is_ok"]:
    if top_integ_report["is_corrupt"]:
        corrupted_names = top_integ_report["corrupt_dbs"]
        st.error(f"🚨 **DATABASE CORRUPTION DETECTED (Fail-Closed):** {', '.join(corrupted_names)}. New trade entries are BLOCKED.")
    elif top_integ_report["is_stale"]:
        stale_names = top_integ_report["stale_dbs"]
        last_str = top_integ_report["latest_utc_dt"].strftime('%H:%M:%S UTC') if top_integ_report.get("latest_utc_dt") else "UNKNOWN"
        st.warning(f"🟠 **DATABASE INTEGRITY SIGNAL STALE:** {', '.join(stale_names)} (last verified {last_str}, daemon not publishing). New trade entries are BLOCKED.")

# ── Top Macro & Intelligence Banner ─────────────────────────────────────────
macro_trend = _macro_regime(query_system_state("btc_macro_trend", ""))
risk_halt_state = _published_flag(query_system_state("RISK_HALT", ""), "no halt state published")
rec_lev_state = _published_state(query_system_state("recommended_leverage", ""), "no recommended leverage published")
fg_num = _optional_float(query_system_state("fear_greed_index", ""))
news_state = _published_flag(query_system_state("news_event_active", ""), "no news gate state published")

if macro_trend == "BULLISH":
    banner_color = "#00ff66"
    banner_bg = "rgba(0, 255, 102, 0.08)"
    banner_icon = "🟢"
    regime_text = "REGIME: MACRO BULLISH (LONG ONLY)"
elif macro_trend == "BEARISH":
    banner_color = "#ff003c"
    banner_bg = "rgba(255, 0, 60, 0.08)"
    banner_icon = "🔴"
    regime_text = "REGIME: MACRO BEARISH (SHORT ONLY)"
elif macro_trend == "NEUTRAL":
    banner_color = "#ff9900"
    banner_bg = "rgba(255, 153, 0, 0.08)"
    banner_icon = "🟠"
    regime_text = "REGIME: RANGE-BOUND CHOP"
else:
    banner_color = "#94a3b8"
    banner_bg = "rgba(148, 163, 184, 0.08)"
    banner_icon = "⚪"
    regime_text = "REGIME: MACRO UNKNOWN (NO STANCE)"

lev_text = f"{html.escape(rec_lev_state['text'])}x Dynamic" if rec_lev_state["measured"] else "not measured"
fg_text = f"{fg_num:.0f}/100" if fg_num is not None else "not measured"
if news_state["measured"]:
    news_text = "EVENT VETO" if news_state["flag"] else "CLEAR"
    news_color = "#ff003c" if news_state["flag"] else "#00ff66"
else:
    news_text = "NOT PUBLISHED"
    news_color = "#94a3b8"
if risk_halt_state["measured"]:
    halted = risk_halt_state["flag"]
    halt_badge = "badge-red" if halted else "badge-green"
    halt_text = "ENTRIES PAUSED (CLOSE ONLY)" if halted else "SYSTEM ACTIVE"
else:
    halt_badge = "badge-neutral"
    halt_text = "HALT STATE NOT PUBLISHED"

st.markdown(f"""
<div style="
    background: {banner_bg};
    border: 1px solid {banner_color};
    border-radius: 6px;
    padding: 12px 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
">
    <div>
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: {banner_color};">
            {banner_icon} {regime_text}
        </span>
        <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 15px;">
            ⚡ Lev: <b style="color: #00e5ff;">{lev_text}</b> | 🧠 F&G: <b style="color: #e2e8f0;">{fg_text}</b> | 📰 Event Gate: <b style="color: {news_color};">{news_text}</b>
        </span>
    </div>
    <div>
        <span class="{halt_badge}">
            {halt_text}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── AI Sovereign Market Posture & Executive Intelligence Banner ───────────────
ai_posture = query_system_state("ai_macro_posture", "TACTICAL_DEFENSE")
ai_summary = query_system_state("ai_macro_summary", "Autonomous macro intelligence apparatus actively monitoring cross-market equity contagion, dollar velocity, and derivatives skew.")
ai_bias = query_system_state("ai_macro_bias", "NEUTRAL_CHOP")
ai_briefing_raw = query_system_state("ai_macro_briefing", "")
ai_risks = []
ai_guidance = ""
if ai_briefing_raw:
    try:
        b_json = json.loads(ai_briefing_raw)
        ai_risks = b_json.get("key_risks", [])
        ai_guidance = b_json.get("tactical_guidance", "")
    except Exception:
        pass

if "AGGRESSIVE" in ai_posture.upper():
    posture_color = "#00ff66"
    posture_bg = "rgba(0, 255, 102, 0.08)"
    posture_icon = "🚀"
elif "PRESERV" in ai_posture.upper() or "DEFENSE" in ai_posture.upper():
    posture_color = "#ff9900"
    posture_bg = "rgba(255, 153, 0, 0.08)"
    posture_icon = "🛡️"
else:
    posture_color = "#00e5ff"
    posture_bg = "rgba(0, 229, 255, 0.08)"
    posture_icon = "🧠"

risks_chips = "".join([f"<span style='background: rgba(255,255,255,0.06); color: #cbd5e1; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; margin-right: 6px;'>⚠️ {r}</span>" for r in ai_risks[:2]])
guidance_html = f"<div style='margin-top: 6px; font-size: 0.76rem; color: #94a3b8;'>🎯 <b>Directive:</b> <span style='color: #00e5ff;'>{ai_guidance}</span></div>" if ai_guidance else ""

st.markdown(f"""
<div style="
    background: {posture_bg};
    border: 1px solid {posture_color};
    border-radius: 6px;
    padding: 10px 18px;
    margin-bottom: 20px;
">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 800; color: {posture_color};">
                {posture_icon} AI SOVEREIGN INTEL: {ai_posture} ({ai_bias})
            </span>
            <span style="color: #94a3b8; font-size: 0.75rem; margin-left: 12px;">Google Gemini 20m Continuous Synthesis</span>
        </div>
        <div>
            {risks_chips}
        </div>
    </div>
    <div style="margin-top: 6px; font-size: 0.82rem; color: #e2e8f0; line-height: 1.45;">
        {ai_summary}
    </div>
    {guidance_html}
</div>
""", unsafe_allow_html=True)

# ── Dynamic Account Sizing & Performance Calculations ────────────────────────
# Fetch All Closed Trades from all fleet trade tables (including swing & super bot maker grid)
df_t1 = query_db(TRADING_DATA_DB, "SELECT *, exposure_time_mins as duration_mins FROM trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t2 = query_db(TRADING_DATA_DB, "SELECT *, exposure_time_mins as duration_mins FROM berserker_trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t3 = query_db(TRADING_DATA_DB, "SELECT *, NULL as duration_mins FROM marl_live_trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t4 = query_db(TRADING_DATA_DB, "SELECT *, NULL as duration_mins FROM mean_reversion_trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t5 = query_db(TRADING_DATA_DB, "SELECT id, timestamp, symbol, side, entry_price, exit_price, size as qty, pnl, notes, 'closed' as status, (holding_hours * 60.0) as duration_mins FROM swing_trades")
trade_dfs = [d for d in [df_t1, df_t2, df_t3, df_t4, df_t5] if not d.empty]
df_all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

if not df_all_trades.empty and 'timestamp' in df_all_trades.columns:
    df_all_trades['dt_sort'] = pd.to_datetime(df_all_trades['timestamp'], utc=True, errors='coerce')
    df_all_trades = df_all_trades.sort_values(by='dt_sort', ascending=False).reset_index(drop=True)

# Ledger columns are REAL-affinity, but daemons can persist text ('N/A', '') that breaks
# numeric comparison and mixed-type sorting on this shared frame.
if not df_all_trades.empty:
    for _numcol in ("pnl", "entry_price", "exit_price", "qty", "duration_mins"):
        if _numcol in df_all_trades.columns:
            df_all_trades[_numcol] = pd.to_numeric(df_all_trades[_numcol], errors="coerce")

total_closed_count = len(df_all_trades)
if not df_all_trades.empty:
    wins_df = df_all_trades[df_all_trades['pnl'].astype(float) > 0]
    losses_df = df_all_trades[df_all_trades['pnl'].astype(float) <= 0]
    win_count = len(wins_df)
    loss_count = len(losses_df)
    # No closed trades means no population to divide by, so there is no rate to publish: the card
    # prints the absence and its cause (see _win_rate_value) instead of a measured 0.0% that was
    # never measured. A fleet that closed trades and won none of them still has a real 0.0% rate.
    win_rate_pct = (win_count / total_closed_count) * 100.0 if total_closed_count > 0 else None
    gross_profit = wins_df['pnl'].astype(float).sum()
    gross_loss = abs(losses_df['pnl'].astype(float).sum())
    # No losing trade means no denominator, so there is no ratio to publish: the card prints the
    # absence and its cause (see _profit_factor_delta) instead of a constant that was never
    # measured. An empty ledger has nothing to measure either.
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    net_realized_pnl = gross_profit - gross_loss
else:
    win_count, loss_count, win_rate_pct, profit_factor, net_realized_pnl = 0, 0, None, None, 0.0

# Active Positions Query across all 5 fleet sub-engines
df_open_1 = query_db(TRADING_DATA_DB, "SELECT * FROM open_positions")
df_open_2 = query_db(TRADING_DATA_DB, "SELECT * FROM berserker_open_positions")
df_open_3 = query_db(TRADING_DATA_DB, "SELECT * FROM marl_live_open_positions")
df_open_4 = query_db(TRADING_DATA_DB, "SELECT * FROM mean_reversion_open_positions")
df_open_5 = query_db(TRADING_DATA_DB, "SELECT symbol, side, entry_price, size AS qty, stop_loss, take_profit, trailing_stop, peak_price, opened_at AS entry_time, leverage, notes AS comments FROM swing_open_positions")

open_dfs = [d for d in [df_open_1, df_open_2, df_open_3, df_open_4, df_open_5] if not d.empty]
if open_dfs:
    df_open = pd.concat(open_dfs, ignore_index=True)
    df_open = df_open.drop_duplicates(subset=['symbol'], keep='last')
else:
    df_open = pd.DataFrame()

# Clean distinct active coins
active_coins_set = set()
live_prices = fetch_live_market_prices()
funding_telemetry_map = fetch_live_position_funding()

_deck_positions = [_position_deck_readings(row, live_prices, funding_telemetry_map)
                   for _, row in df_open.iterrows()]
_deck_totals = _fleet_position_rollup(_deck_positions)
total_margin_locked = _deck_totals["margin_locked"]
total_unrealized_pnl = _deck_totals["unrealized_pnl"]
# C-4: an unread funding is not a zero to add. The fleet total is a reading only when every
# position reported one, and an absence from then on.
total_live_funding = _deck_funding_total([p["funding"] for p in _deck_positions])

if not df_open.empty:
    for p in _deck_positions:
        active_coins_set.add(p["clean_symbol"])

max_coin_slots = int(num(query_system_state("GLOBAL_MAX_POSITIONS", "5"), 5.0))
margin_limit_pct = state_num("GLOBAL_MARGIN_LIMIT", 70.0)
bal_for_calc = live_bal_val if live_bal_val is not None else 0.0
_margin_gauge = _deck_margin_gauge(total_margin_locked,
                                  _deck_account_balance(live_bal_val),
                                  _deck_reading(margin_limit_pct))
margin_util_pct = _margin_gauge["utilization"]
margin_bar_width = _margin_gauge["bar_width"]
free_cash_buffer = _margin_gauge["buffer"]
_margin_rung = _deck_margin_rung(margin_util_pct)
margin_color = _margin_rung["color"]
margin_status = _margin_rung["status"]

# ── Hero KPI Cards Grid ──────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    help_deposit = (f"Deposited Capital: {capital['deposited_usdt']} ({capital['tranches']}). "
                    f"True Net Profit: {_profit_display(total_profit, total_roi_pct)}.")
    eq_val_str = f"${live_bal_val:.2f} USDT" if live_bal_val is not None else "Syncing..."
    st.metric(
        label="💰 Live Account Equity",
        value=eq_val_str,
        delta=f"{'+$' if total_profit>=0 else '-$'}{abs(total_profit):.2f} ({total_roi_pct:+.2f}% ROI)" if total_profit is not None else None,
        help=help_deposit
    )
    principal_caption = f"Principal: **{capital['deposited']}** (`{capital['tranches_plain']}`)"
    st.caption(principal_caption)

with kpi2:
    st.metric(
        label="🏆 Fleet Win Rate & PF",
        value=_win_rate_value(win_rate_pct, total_closed_count),
        delta=_profit_factor_delta(profit_factor, win_count, loss_count),
        # An absent ratio is not a fall from the last reading, so it must not wear the green
        # up-arrow st.metric gives every delta that does not start with "-" (streamlit's
        # elements/metric.py::_is_negative_delta). Gray, no arrow: the card says what it could
        # not measure, and nothing more.
        delta_color="off" if profit_factor is None else "normal",
        delta_arrow="off" if profit_factor is None else "auto",
    )

with kpi3:
    st.metric(
        label="🔥 Portfolio Heat & Slots",
        value=f"{len(active_coins_set)} / {max_coin_slots} Slots Used",
        delta=f"{max(0, max_coin_slots - len(active_coins_set))} Slots Open (2 Core / 3 Alt)"
    )

with kpi4:
    _fund_help = (f"Cumulative Accrued Funding Fee: {total_live_funding['value']:+,.4f} USDT"
                  if not _deck_is_absent(total_live_funding) else
                  f"Cumulative Accrued Funding Fee: {_DECK_ABSENT_TEXT} — "
                  f"{total_live_funding['reason']}")
    _unreal_help = (f"Unrealized Mark PnL: {_DECK_ABSENT_TEXT} — {total_unrealized_pnl['reason']}"
                    if _deck_is_absent(total_unrealized_pnl) else
                    f"Unrealized Mark PnL: ${total_unrealized_pnl['value']:+,.4f} USDT")
    st.metric(
        label="⚡ Live Unrealized PnL",
        value=f"{_deck_signed_money(total_unrealized_pnl, 2)} USDT",
        delta=f"Funding: {_deck_funding_text(total_live_funding)}",
        # An unmeasured funding is not a flat one, so it may not arrive as a delta with an
        # up-arrow: grey and no arrow, the kpi2 precedent.
        delta_color="off" if _deck_is_absent(total_live_funding) else "normal",
        delta_arrow="off" if _deck_is_absent(total_live_funding) else "auto",
        help=f"{_unreal_help} | {_fund_help}"
    )

# P0: Global Margin Utilization Visual Progress Bar
fund_badge_color = _deck_funding_color(total_live_funding)
margin_util_display = _deck_text(margin_util_pct, "{:.1f}%")
margin_locked_display = _deck_text(total_margin_locked, "${:,.2f}")
free_cash_display = _deck_text(free_cash_buffer, "${:,.2f}")
st.markdown(f"""
<div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <span style="font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">
            🛡️ Portfolio Margin Utilization: <b style="color: {margin_color};">{margin_util_display}</b>{_deck_note(margin_util_pct)} of {margin_limit_pct:.0f}% Hard Ceiling ({margin_status})
        </span>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #e2e8f0;">
            Margin Locked: <b style="color: {margin_color};">{margin_locked_display}</b>{_deck_note(total_margin_locked)} | Liquid Buffer: <b style="color: {_deck_color(free_cash_buffer, "#00e5ff")};">{free_cash_display} USDT</b>{_deck_note(free_cash_buffer)} | Active Funding: <b style="color: {fund_badge_color};">{_deck_funding_text(total_live_funding)}</b>{_deck_note(total_live_funding)}
        </span>
    </div>
    <div style="width: 100%; background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; position: relative;">
        <div style="width: {_deck_value_or(margin_bar_width):.1f}%; background: {margin_color}; height: 100%; border-radius: 5px; transition: width 0.4s ease-in-out;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabbed Navigation Deck ──────────────────────────────────────────────
tab_briefing, tab_live, tab_ledger, tab_ai, tab_risk, tab_system = st.tabs([
    "📋 5-MIN EXECUTIVE BRIEFING",
    "🎯 LIVE FLEET & RADAR",
    "📜 CLOSED TRADES & LEDGER",
    "🧠 AI BOARDROOM & MEMORY",
    "🛡️ RISK & MICROSTRUCTURE",
    "⚙️ SYSTEM HEALTH & DAEMONS"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0: 5-MINUTE AUTONOMOUS EXECUTIVE BRIEFING
# ═════════════════════════════════════════════════════════════════════════════
with tab_briefing:
    # 1. Fetch current briefing payload
    briefing_raw = query_system_state("executive_briefing_5m", "{}")
    briefing_data = {}
    try:
        briefing_data = json.loads(briefing_raw)
    except Exception:
        pass

    now_ts = time.time()
    last_gen_ts = num(briefing_data.get("generated_at_timestamp"), 0.0)
    age_seconds = now_ts - last_gen_ts if last_gen_ts > 0 else 999999

    # Top Control Bar
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.subheader("📋 5-Minute Executive Briefing & Fleet Telemetry")
        if age_seconds < 300:
            mins_ago = int(age_seconds // 60)
            secs_ago = int(age_seconds % 60)
            age_str = f"{mins_ago}m {secs_ago}s ago" if mins_ago > 0 else f"{secs_ago}s ago"
            next_in = max(0, 300 - int(age_seconds))
            next_mins = next_in // 60
            next_secs = next_in % 60
            st.markdown(
                f"<span style='background: rgba(0, 255, 102, 0.15); color: #00ff66; padding: 5px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; font-size: 0.85rem;'>"
                f"🟢 AUTO-UPDATING EVERY 5 MINS | Cycle #{briefing_data.get('cycle_num', 1)} | Generated {age_str} | Next update in {next_mins}m {next_secs:02d}s"
                f"</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<span style='background: rgba(255, 153, 0, 0.15); color: #ff9900; padding: 5px 12px; border-radius: 4px; font-weight: bold; font-family: monospace; font-size: 0.85rem;'>"
                "🟡 REFRESHING BRIEFING | Synthesizing latest 5-minute cycle..."
                "</span>",
                unsafe_allow_html=True
            )

    with top_c2:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh", width="stretch", help="Immediately runs a fresh 5-minute briefing cycle across live exchange, open positions, AI ledger, and daemons."):
                if BRIEFING_ENGINE_AVAILABLE:
                    curr_c = briefing_data.get("cycle_num", 0) + 1
                    briefing_data = run_single_briefing_cycle(curr_c)
                    st.toast("✅ Executive Briefing regenerated successfully!")
                    st.rerun()
        with col_btn2:
            if st.button("📧 Send 1h Email", width="stretch", help="Immediately synthesizes the last 60 minutes of 5-minute briefings and sends the 1-Hour Executive Briefing email via Gmail SMTP."):
                try:
                    import hourly_briefing
                    import importlib
                    importlib.reload(hourly_briefing)
                    with st.spinner("Dispatching 1h Email..."):
                        sent_ok = hourly_briefing.run_single_hourly_briefing(force=True)
                        if sent_ok:
                            st.toast("🏆 1-Hour Executive Rollup Email dispatched!")
                            st.success("🏆 1-Hour Executive Rollup Email sent to your inbox!")
                        else:
                            st.error("Email dispatch failed. Check SMTP.")
                except Exception as em_t0_err:
                    st.error(f"Error: {em_t0_err}")

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

    # 4 Executive KPI Metric Cards
    b_eq = num(briefing_data.get("equity"), bal_for_calc)
    b_margin_stored = _optional_float(briefing_data.get("margin_locked"))
    b_margin = _deck_reading(b_margin_stored) if b_margin_stored is not None else total_margin_locked
    b_margin_display = _deck_text(b_margin, "${:,.2f}")
    b_util_pct = num(briefing_data.get("margin_util_pct"))
    b_slots = int(num(briefing_data.get("slots_occupied"), len(df_open)))
    b_rules = int(num(briefing_data.get("rules_count"), 108))
    max_slots_calc = max(4, max_coin_slots)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="💰 Live Bitget Equity",
            value=f"${b_eq:,.2f} USDT",
            delta=f"+${b_eq - total_deposited:.2f} ({((b_eq - total_deposited)/total_deposited)*100:.1f}%)" if total_deposited is not None and total_deposited > 0 else None,
            delta_color="normal"
        )
    with kpi2:
        st.metric(
            label="🛡️ Margin Utilized (70% Max)",
            value=f"{b_util_pct:.1f}%",
            delta=f"{b_margin_display} locked (Ceiling: 70%)",
            delta_color="inverse" if b_util_pct > 60 else "normal"
        )
    with kpi3:
        st.metric(
            label="🎯 Fleet Active Slots",
            value=f"{b_slots} / {max_slots_calc} Slots",
            delta=f"{max(0, max_slots_calc - b_slots)} Slots Open",
            delta_color="normal"
        )
    with kpi4:
        st.metric(
            label="🧠 Conviction Boardroom",
            value=f"{b_rules} Rules",
            delta="Continuous Forensic Autopsies",
            delta_color="normal"
        )

    st.markdown("---")

    # Render AI Sovereign Commander's Tactical Memo Dispatch Box
    memo_raw = query_system_state("ai_commander_memo", "")
    if memo_raw:
        try:
            memo_data = json.loads(memo_raw)
            if isinstance(memo_data, dict) and memo_data.get("headline"):
                watch_str = ", ".join(memo_data.get("watch_levels", []))
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%); 
                                border: 1px solid #3b82f6; border-left: 5px solid #3b82f6; border-radius: 8px; 
                                padding: 18px 22px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 0.9rem; font-weight: 700; color: #60a5fa; text-transform: uppercase; letter-spacing: 0.05em;">
                                🎖️ AI Sovereign Commander's Tactical Memo
                            </span>
                            <span style="font-size: 0.75rem; color: #94a3b8; font-family: monospace;">
                                {memo_data.get('updated_at', '')[:19]} UTC
                            </span>
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 10px;">
                            ⚡ {memo_data.get('headline')}
                        </div>
                        <div style="font-size: 0.92rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 8px;">
                            <strong>Positioning Rationale:</strong> {memo_data.get('positioning_rationale')}
                        </div>
                        <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.4; margin-bottom: 8px;">
                            <strong>Setup Screening & Vetoes:</strong> {memo_data.get('rejection_summary')}
                        </div>
                        <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
                            <span style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); color: #93c5fd; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-family: monospace;">
                                🎯 Watch: {watch_str}
                            </span>
                            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #6ee7b7; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem;">
                                🧭 Stance: {memo_data.get('strategic_directive')}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception:
            pass

    # Render Main Briefing Markdown Card in styled container
    md_content = briefing_data.get("markdown") or query_system_state("executive_briefing_markdown", "")
    if md_content:
        st.markdown(
            f"""
            <div style="background: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 22px; box-shadow: 0 4px 14px rgba(0,0,0,0.4);">
            """,
            unsafe_allow_html=True
        )
        st.markdown(md_content)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Generating first briefing cycle... Click 'Force Briefing Refresh' if not loaded.")

    st.markdown("---")

    # History Archive Expander
    with st.expander("📜 Historical Briefings Archive (5-Minute Cycles & 1-Hour Email Rollups)", expanded=False):
        st.markdown("##### 🏆 Dispatched 1-Hour Executive Email Rollups")
        hourly_hist = query_db(
            TRADING_LEDGER_DB,
            "SELECT id as 'ID', generated_at as 'Dispatched At (UTC)', start_cycle as 'From Cycle', end_cycle as 'To Cycle', printf('$%.2f', start_equity) as 'Start Equity', printf('$%.2f', end_equity) as 'End Equity', printf('%+0.2f', equity_delta) as 'Delta ($)', slots_occupied as 'Slots', dominant_regime as 'Regime' FROM hourly_briefing_history ORDER BY id DESC LIMIT 10"
        )
        if not hourly_hist.empty:
            st.dataframe(hourly_hist, width="stretch", hide_index=True)
        else:
            st.caption("No 1-hour email rollups dispatched yet.")

        st.markdown("---")
        st.markdown("##### ⏱️ Archived 5-Minute Briefing Cycles")
        hist_rows = query_db(
            TRADING_LEDGER_DB,
            "SELECT id, cycle_num, generated_at, equity, margin_util_pct, slots_occupied, regime, markdown FROM executive_briefing_history ORDER BY id DESC LIMIT 10"
        )
        if not hist_rows.empty:
            cycle_options = [
                f"Cycle #{r['cycle_num']} — {r['generated_at']} | Equity: ${num(r['equity']):,.2f} | Slots: {r['slots_occupied']} | {r['regime']}"
                for _, r in hist_rows.iterrows()
            ]
            selected_cycle_label = st.selectbox("Select Past Briefing Cycle to Review:", cycle_options, index=0)
            sel_idx = cycle_options.index(selected_cycle_label)
            selected_markdown = hist_rows.iloc[sel_idx]["markdown"]
            st.markdown(f"##### 📜 Archived Briefing ({selected_cycle_label})")
            st.markdown(selected_markdown)
        else:
            st.caption("No historical briefings stored yet. Subsequent cycles will archive here.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE FLEET & WATCHLIST RADAR (Dual-Layer Stops + Liquidation + Age)
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🎯 Active Live Positions (Real-Time Price & Risk Radar)")
    
    if not df_open.empty:
        # Fleet Accrued Funding Summary Banner
        total_fund_color = _deck_funding_color(total_live_funding)
        # C-4 decision: SUPPRESS, never label. A "True Net (Mark + Funding)" that quietly left an
        # unmeasured funding out would still publish under a name that says the funding is in it.
        true_portfolio_net = _deck_sum(total_unrealized_pnl, total_live_funding)
        true_port_color = _deck_pnl_color(true_portfolio_net)

        st.markdown(f"""
        <div style="background: rgba(13, 18, 31, 0.85); border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="color: #94a3b8; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.05em;">Live Portfolio Funding & Carry Telemetry</span><br>
                <span style="font-size: 0.95rem; font-weight: bold; color: #ffffff;">Fleet Accrued Funding: </span>
                <span style="font-size: 1.05rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {total_fund_color};">{_deck_funding_text(total_live_funding)}</span>{_deck_note(total_live_funding)}
            </div>
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">
                <span style="color: #94a3b8;">Unrealized Mark PnL: <b style="color: {_deck_pnl_color(total_unrealized_pnl)};">{_deck_signed_money(total_unrealized_pnl, 2)}</b>{_deck_note(total_unrealized_pnl)}</span><br>
                <span style="color: #94a3b8;">True Net (Mark + Funding): <b style="color: {true_port_color};">{_deck_signed_money(true_portfolio_net, 2)} USDT</b>{_deck_note(true_portfolio_net)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render visual metric cards for each live position
        pos_cols = st.columns(len(df_open))
        display_pos = []
        
        for idx, row in df_open.iterrows():
            p_read = _position_deck_readings(row, live_prices, funding_telemetry_map)
            sym = p_read["symbol"]
            sym_clean = p_read["clean_symbol"]
            side = p_read["side"]
            hard_sl = p_read["hard_sl"]
            qty = num(row.get('qty'))
            entry_p = num(row.get('entry_price'))
            sl = num(row.get('stop_loss'))
            tp = num(row.get('take_profit'))
            lev = p_read["leverage"]
            margin = p_read["margin"]
            mark = p_read["mark"]
            entry_time_raw = row.get('entry_time')
            pos_age = compute_position_age(entry_time_raw)
            sl_breach_raw = row.get('sl_breach_time')
            has_flash_breach = bool(sl_breach_raw and str(sl_breach_raw).strip() not in ("", "None", "0", "N/A"))

            # Liquidation estimation (panel's own model; see _position_liquidation)
            est_liq = p_read["est_liq"]
            dist_liq_pct = p_read["dist_liq"]

            # Unrealized PnL & ROI
            unr_pnl = p_read["unr_pnl"]
            unr_roi = p_read["unr_roi"]

            # Live Position Funding Telemetry: the reading the map gives for this symbol, or an
            # absence naming the map it could not read.
            pos_funding = p_read["funding"]
            pos_funding_str = _deck_funding_text(pos_funding)
            pos_fund_color = _deck_funding_color(pos_funding)
            pos_funding_badge = (f"<b style='color: {pos_fund_color};'>{pos_funding_str}</b>"
                                 f"{_deck_note(pos_funding)}")

            # True Net PnL (Mark PnL + Funding). C-4 decision: SUPPRESSED when either addend is
            # unmeasured, not published as "mark only" — the label would still claim the netting.
            true_net_pnl = _deck_sum(unr_pnl, pos_funding)
            true_net_roi = _position_roi(true_net_pnl, margin)
            true_net_str = (f"{_deck_signed_money(true_net_pnl)} "
                            f"({_deck_text(true_net_roi, '{:+.2f}%')}{_deck_note(true_net_roi)})")
            true_net_color = _deck_color(true_net_roi, _deck_pnl_color(true_net_pnl))
            
            # Target Distances
            if side == "long":
                is_risk_free = (sl >= entry_p)
            else:
                is_risk_free = (sl <= entry_p and sl > 0)
            tp_str = (_position_target_label(p_read["dist_tp"], side, "tp")
                      if tp > 0 else "Active Trail")
            sl_str = (_position_target_label(p_read["dist_sl"], side, "sl")
                      if sl > 0 else "Active Trail")

            with pos_cols[idx % len(pos_cols)]:
                pnl_color = _deck_pnl_color(unr_pnl)
                ai_runner_status = query_system_state(f"ai_runner_{sym_clean}", "")
                is_runner = (bool(row.get('runner_mode', False)) if hasattr(row, 'get') else False) or (ai_runner_status == "RUNNER_MODE")
                is_harvest = (bool(row.get('harvest_mode', False)) if hasattr(row, 'get') else False) or (ai_runner_status == "HARVEST_MODE")
                runner_badge = "<span style='background: rgba(0,229,255,0.18); color: #00e5ff; border: 1px solid #00e5ff; font-size: 0.68rem; padding: 1px 5px; border-radius: 3px; font-weight: bold; margin-left: 4px;'>🏃 RUNNER</span>" if is_runner else ("<span style='background: rgba(255,153,0,0.18); color: #ff9900; border: 1px solid #ff9900; font-size: 0.68rem; padding: 1px 5px; border-radius: 3px; font-weight: bold; margin-left: 4px;'>🌾 HARVEST</span>" if is_harvest else "")
                risk_free_badge = "<span style='background: rgba(0,255,102,0.15); color: #00ff66; border: 1px solid #00ff66; font-size: 0.68rem; padding: 1px 5px; border-radius: 3px; font-weight: bold;'>🛡️ RISK-FREE</span>" if is_risk_free else ""
                is_scalp_short = (side == 'short') or ("scalp_short" in str(row.get('comments', '')).lower())
                scalp_short_badge = "<span style='background: rgba(244,63,94,0.18); color: #f43f5e; border: 1px solid #f43f5e; font-size: 0.68rem; padding: 1px 5px; border-radius: 3px; font-weight: bold; margin-left: 4px;'>⚡ QUICK SCALP (45m Max)</span>" if is_scalp_short else ""
                flash_timer_badge = "<div style='margin-top: 4px; background: rgba(255,0,60,0.2); border: 1px solid #ff003c; border-radius: 4px; padding: 2px 6px; font-size: 0.72rem; color: #ff003c; font-weight: bold;'>⚠️ FLASH CRASH TIMER RUNNING</div>" if has_flash_breach else ""

                # Query AI Sovereign Position Health Check
                ai_health_raw = query_system_state(f"ai_health_{sym_clean}", "")
                ai_health_badge = ""
                if ai_health_raw:
                    try:
                        h_obj = json.loads(ai_health_raw)
                        chk_time_str = h_obj.get("checked_at")
                        is_fresh = True
                        if chk_time_str:
                            try:
                                chk_dt = datetime.fromisoformat(chk_time_str.replace("Z", "+00:00"))
                                age_mins = (datetime.now(timezone.utc) - chk_dt).total_seconds() / 60.0
                                if age_mins > 60.0:
                                    is_fresh = False
                            except Exception:
                                pass
                        if is_fresh:
                            h_verdict = h_obj.get("verdict", "HOLD")
                            h_grade = h_obj.get("health_grade", "PRISTINE")
                            h_diag = h_obj.get("diagnosis", "")
                            h_color = "#00ff66" if h_grade == "PRISTINE" else ("#ff9900" if h_grade == "WARNING" else "#ff003c")
                            ai_health_badge = f"<div style='margin-top: 5px; background: rgba(0,0,0,0.35); border-left: 2px solid {h_color}; padding: 3px 6px; border-radius: 3px; font-size: 0.72rem; color: #cbd5e1;'>🩺 <b>AI Health ({h_grade}):</b> <span style='color: {h_color}; font-weight: bold;'>{h_verdict}</span> — {h_diag}</div>"
                    except Exception:
                        pass

                # Query AI Shepherd Verdict & In-Flight Reversal/Carry Sentinel
                shep_raw = query_system_state(f"ai_shepherd_{sym_clean}", "")
                shep_badge = ""
                shep_verdict_str = "MONITORED"
                if shep_raw:
                    try:
                        s_obj = json.loads(shep_raw)
                        s_verd = s_obj.get("verdict", "HOLDING_NORMAL")
                        s_diag = s_obj.get("diagnosis", "")
                        s_rat = bool(s_obj.get("sl_ratcheted", False))
                        s_col = "#00e5ff" if "RUNNER" in s_verd else ("#00ff66" if s_rat or "BREAKEVEN" in s_diag.upper() else "#94a3b8")
                        shep_verdict_str = s_verd
                        shep_badge = f"<div style='margin-top: 5px; background: rgba(0, 229, 255, 0.06); border-left: 2px solid {s_col}; padding: 3px 6px; border-radius: 3px; font-size: 0.72rem; color: #cbd5e1;'>🦅 <b>In-Flight Sentinel:</b> <span style='color: {s_col}; font-weight: bold;'>{s_verd}</span> {('⚡ SL RATCHETED' if s_rat else '')} — <span style='color: #94a3b8;'>{s_diag[:65]}...</span></div>"
                    except Exception:
                        pass

                card_html = (
                    f'<div class="pos-card" style="border-left: 3px solid {pnl_color};">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div><b style="font-size: 1.1rem; color: #ffffff;">{sym}</b> {risk_free_badge}{runner_badge}{scalp_short_badge}</div>'
                    f'<span class="{"badge-green" if side=="long" else "badge-red"}">{side.upper()} {_deck_label(lev, "leverage", "{:.0f}x")}</span>'
                    f'</div>'
                    f'<div style="margin: 8px 0; font-family: monospace; font-size: 1.25rem; font-weight: bold; color: {pnl_color};">'
                    f'{_deck_signed_money(unr_pnl)} ({_deck_text(unr_roi, "{:+.2f}%")}{_deck_note(unr_roi)})'
                    f'</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">'
                    f'Entry: <b>${entry_p:,.4f}</b> | Mark: <b style="color: {_deck_color(mark, "#00e5ff")};">{_deck_text(mark, "${:,.4f}")}</b>{_deck_note(mark)}<br>'
                    f'⏱️ Age: <b style="color: #e2e8f0;">{pos_age}</b> | Locked: <b style="color: {_deck_color(margin, "#e2e8f0")};">{_deck_text(margin, "${:,.2f}")}</b>{_deck_note(margin)}<br>'
                    f'💸 Live Funding: {pos_funding_badge}<br>'
                    f'📊 True Net PnL: <b style="color: {true_net_color};">{true_net_str}</b><br>'
                    f'🎯 Target: <b style="color: #00ff66;">{tp_str} (${tp:,.4f})</b><br>'
                    f'🛡️ Soft SL: <b style="color: #ff9900;">${sl:,.4f} ({sl_str})</b><br>'
                    f'🧱 Hard SL: <b style="color: #94a3b8;">{_deck_text(hard_sl, "${:,.4f}")}</b>{_deck_note(hard_sl)}<br>'
                    f'☠️ Est. Liq (panel model, {_PANEL_LIQ_MODEL_MMR*100:.1f}% assumed MMR): <b style="color: {_deck_color(est_liq, "#ff003c")};">{_deck_text(est_liq, "${:,.4f}")}</b>{_deck_note(est_liq)} | Safety buffer: <b style="color: {_deck_color(dist_liq_pct, "#ff003c")};">{_deck_text(dist_liq_pct, "+{:.1f}%")}</b>{_deck_note(dist_liq_pct)}'
                    f'{flash_timer_badge}'
                    f'{ai_health_badge}'
                    f'{shep_badge}'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            
            display_pos.append({
                "Asset": sym,
                "Side": side.upper(),
                "Leverage": _deck_label(lev, "leverage", "{:.0f}x"),
                "Size": f"{qty:.4f}",
                "Entry Price": f"${entry_p:,.4f}",
                "Live Price": _deck_text(mark, "${:,.4f}"),
                "Unrealized PnL": _deck_signed_money(unr_pnl),
                "Funding (Live)": pos_funding_str,
                "True Net PnL": _deck_signed_money(true_net_pnl),
                "ROI (%)": _deck_text(unr_roi, "{:+.2f}%"),
                "Soft SL": f"${sl:,.4f}",
                "Hard SL (recorded)": _deck_text(hard_sl, "${:,.4f}"),
                "Est. Liq Price (panel model)": _deck_text(est_liq, "${:,.4f}"),
                "Liq Safety Buffer (panel model)": _deck_text(dist_liq_pct, "+{:.1f}%"),
                "In-Flight Sentinel": shep_verdict_str,
                "Age": pos_age,
                "Margin Locked": _deck_text(margin, "${:,.2f}"),
                "Status": "🟢 RISK-FREE" if is_risk_free else "🟢 LIVE EXECUTED"
            })
            
        st.markdown("#### 📋 Detailed Active Position & Risk Matrix")
        st.dataframe(pd.DataFrame(display_pos), width="stretch", hide_index=True)

        # ── 1.2 Live Position Execution Charts (Active Fleet Only) ───────────────
        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        st.subheader("📈 Live Position Execution Charts (Active Positions Only)")

        chart_c1, chart_c2, chart_c3 = st.columns([1.5, 1.5, 3])
        with chart_c1:
            chart_tf = st.selectbox(
                "Chart Timeframe",
                options=["5m", "15m", "1h", "4h"],
                index=1,
                help="Select candle timeframe for active position execution charts.",
                key="pos_chart_tf"
            )
        with chart_c2:
            chart_limit = st.selectbox(
                "Candles Lookback",
                options=[30, 50, 100],
                index=1,
                help="Number of historical candles to render on chart.",
                key="pos_chart_limit"
            )
        with chart_c3:
            show_emas = st.checkbox(
                "Show Trend EMAs (EMA 9 Fast / EMA 21 Trend)",
                value=True,
                help="Overlays EMA 9 (yellow) and EMA 21 (cyan) on the price action.",
                key="pos_chart_emas"
            )

        # Build dynamic tabs: one tab for each coin currently open + an All-in-One view
        open_pos_list = []
        for idx, row in df_open.iterrows():
            p_read = _position_deck_readings(row, live_prices, funding_telemetry_map)
            s_name = str(row['symbol'])
            s_clean = s_name.upper().replace("/", "").replace("-", "").split(":")[0]
            s_side = str(row.get('side') or 'long').upper()
            s_entry = num(row.get('entry_price'))
            s_sl = num(row.get('stop_loss'))
            s_tp = num(row.get('take_profit'))
            s_curr = p_read["mark"]
            s_lev = p_read["leverage"]
            s_qty = num(row.get('qty'))
            s_margin = p_read["margin"]
            s_unr = p_read["unr_pnl"]
            s_roi = p_read["unr_roi"]
            s_liq = p_read["est_liq"]

            open_pos_list.append({
                "symbol": s_name,
                "clean_symbol": s_clean,
                "side": s_side,
                "entry": s_entry,
                "sl": s_sl,
                "hard_sl": p_read["hard_sl"],
                "tp": s_tp,
                "curr": s_curr,
                "leverage": s_lev,
                "qty": s_qty,
                "margin": s_margin,
                "unr_pnl": s_unr,
                "unr_roi": s_roi,
                "est_liq": s_liq
            })

        chart_tab_titles = [
            f"📈 {p['symbol']} ({_deck_signed_money(p['unr_pnl'], 2)})"
            for p in open_pos_list
        ]
        if len(open_pos_list) > 1:
            chart_tab_titles.append("📊 All Active Fleet Charts (Stacked)")

        pos_chart_tabs = st.tabs(chart_tab_titles)

        for i, p in enumerate(open_pos_list):
            with pos_chart_tabs[i]:
                # Top summary pill for this specific position
                pnl_color = _deck_pnl_color(p["unr_pnl"])
                st.markdown(f"""
                <div style="background: rgba(13, 18, 31, 0.7); border: 1px solid #1e293b; border-left: 3px solid {pnl_color}; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <b style="font-size: 1.05rem; color: #ffffff;">{p['symbol']}</b>
                        <span style="background: rgba(0, 229, 255, 0.15); color: #00e5ff; font-size: 0.75rem; padding: 2px 6px; border-radius: 3px; font-weight: bold; margin-left: 6px;">{p['side']} {_deck_label(p['leverage'], "leverage", "{:.0f}x")}</span>
                        <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 10px;">Size: <b style="color: #e2e8f0;">{p['qty']}</b> ({_deck_text(p['margin'], "${:,.2f}")} locked)</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                        <span style="color: #94a3b8;">Entry: <b style="color: #00e5ff;">${p['entry']:,.4f}</b></span> | 
                        <span style="color: #94a3b8;">Mark: <b style="color: {_deck_color(p['curr'], "#ffffff")};">{_deck_text(p['curr'], "${:,.4f}")}</b>{_deck_note(p['curr'])}</span> | 
                        <span style="color: #94a3b8;">Unrealized PnL: <b style="color: {pnl_color};">{_deck_signed_money(p['unr_pnl'])} ({_deck_text(p['unr_roi'], '{:+.2f}%')}{_deck_note(p['unr_roi'])})</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = build_position_candlestick_chart(
                    symbol=p["symbol"],
                    side=p["side"].lower(),
                    entry_p=p["entry"],
                    sl=p["sl"],
                    hard_sl=_deck_chart_level(p["hard_sl"]),
                    tp=p["tp"],
                    curr_p=_deck_chart_level(p["curr"]),
                    est_liq=_deck_chart_level(p["est_liq"]),
                    timeframe=chart_tf,
                    limit=chart_limit,
                    show_emas=show_emas
                )
                if fig:
                    st.plotly_chart(fig, width="stretch", key=f"fig_{p['clean_symbol']}_{chart_tf}")
                else:
                    st.warning(f"Could not load live candle stream for {p['symbol']}. Check exchange connectivity.")

        if len(open_pos_list) > 1:
            with pos_chart_tabs[-1]:
                st.markdown("##### 📊 Fleet-Wide Active Position Visual Radar")
                for p in open_pos_list:
                    pnl_color = _deck_pnl_color(p["unr_pnl"])
                    st.markdown(f"**{p['symbol']} ({p['side']} {_deck_label(p['leverage'], 'leverage', '{:.0f}x')})** — Entry: `${p['entry']:,.4f}` | Mark: `{_deck_text(p['curr'], '${:,.4f}')}` | PnL: <span style='color: {pnl_color}; font-weight: bold;'>{_deck_signed_money(p['unr_pnl'])}</span>{_deck_note(p['curr'])}", unsafe_allow_html=True)
                    fig_all = build_position_candlestick_chart(
                        symbol=p["symbol"],
                        side=p["side"].lower(),
                        entry_p=p["entry"],
                        sl=p["sl"],
                        hard_sl=_deck_chart_level(p["hard_sl"]),
                        tp=p["tp"],
                        curr_p=_deck_chart_level(p["curr"]),
                        est_liq=_deck_chart_level(p["est_liq"]),
                        timeframe=chart_tf,
                        limit=chart_limit,
                        show_emas=show_emas
                    )
                    if fig_all:
                        st.plotly_chart(fig_all, width="stretch", key=f"fig_all_{p['clean_symbol']}_{chart_tf}")

        # ── 1.3 Continuous AI In-Flight Position Shepherd (15-Minute Scans) ───────
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("🦅 AI In-Flight Position Shepherd (15-Minute L2 Tape & Runner Sentinel)")
        
        shepherd_cards = []
        for p_row in df_open.itertuples():
            sym_raw = str(p_row.symbol)
            clean_s = sym_raw.replace("/", "").replace("-", "").upper()
            raw_shep = query_system_state(f"ai_shepherd_{clean_s}", "")
            if raw_shep:
                try:
                    s_data = json.loads(raw_shep)
                    shepherd_cards.append((sym_raw, s_data))
                except Exception:
                    pass

        if shepherd_cards:
            cols = st.columns(min(3, len(shepherd_cards)))
            for idx, (sym_name, s_data) in enumerate(shepherd_cards):
                c_col = cols[idx % len(cols)]
                with c_col:
                    v_val = s_data.get("verdict", "HOLDING_NORMAL")
                    is_run = s_data.get("runner_mode", False)
                    v_color = "#00e5ff" if is_run else ("#00ff66" if v_val == "HOLDING_NORMAL" else ("#ff9900" if "STALL" in v_val else "#ff003c"))
                    v_icon = "🏃" if is_run else ("🛡️" if v_val == "HOLDING_NORMAL" else "⚠️")
                    r_mult = num(s_data.get("r_multiple"), 0.0)
                    tgt_r = num(s_data.get("target_r"), 2.5)
                    tape = s_data.get("tape_condition", "BALANCED")
                    diag = s_data.get("diagnosis", "Position monitored.")
                    sl_sugg = s_data.get("suggested_sl")
                    sl_text = f"${float(sl_sugg):,.4f}" if sl_sugg else "Hold Existing"
                    ratchet_badge = " <span style='color: #00ff66; font-size: 0.75rem; font-weight: bold;'>[RATCHETED]</span>" if s_data.get("sl_ratcheted") else ""

                    st.markdown(f"""
                    <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {v_color}; padding: 12px; border-radius: 6px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <b style="color: #ffffff; font-size: 1.05rem;">{sym_name}</b>
                            <span style="background: rgba(0, 229, 255, 0.15); color: {v_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.78rem;">
                                {v_icon} {v_val}
                            </span>
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin-bottom: 6px;">
                            Progress: <b style="color: #00ff66;">{r_mult:+.2f}R</b> (Target: <b style="color: #00e5ff;">{tgt_r:.1f}R</b>)<br>
                            L2 Tape: <b style="color: #e2e8f0;">{tape}</b> | Suggested SL: <b style="color: #ff9900;">{sl_text}</b>{ratchet_badge}
                        </div>
                        <div style="font-size: 0.78rem; color: #cbd5e1; background: rgba(255,255,255,0.02); padding: 8px; border-radius: 4px; border-left: 2px solid rgba(255,255,255,0.1);">
                            <i>"{diag}"</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Continuous In-Flight AI Shepherd evaluates open positions every 15 minutes feeding live candle structure and L2 order book tape.")
    else:
        # ── 1.1b Autonomous Capital Preservation & Dual-Regime Scanning Deck ──────
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(13, 18, 31, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); 
                    border: 1px solid #1e293b; border-left: 4px solid #00ff66; border-radius: 8px; 
                    padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(0,0,0,0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
                <div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: #00ff66;">
                        🛡️ 100% CAPITAL PRESERVED — ZERO NAKED RISK
                    </span>
                    <span style="color: #94a3b8; font-size: 0.82rem; margin-left: 12px;">
                        Liquid Buffer: <b style="color: #38bdf8;">{bal_display} USDT</b> | Margin Utilized: <b style="color: #00ff66;">0.0%</b> (70% Max Ceiling)
                    </span>
                </div>
                <div>
                    <span style="background: rgba(0, 255, 102, 0.15); color: #00ff66; border: 1px solid #00ff66; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 0.80rem;">
                        🟢 CASH PRESERVATION ACTIVE
                    </span>
                </div>
            </div>
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.5; margin-bottom: 10px;">
                All portfolio capital is safely parked in liquid USDT. The autonomous engine has suppressed counter-trend long entries during the current market posture (<b>Anti-Dead-Cat-Bounce Gate</b> active). Zero AI calls wasted; 100% free tier credits preserved.
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; font-size: 0.80rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
                <div>
                    <b style="color: #00e5ff;">🤖 4H Swing Architect (Longs):</b><br>
                    <span style="color: #94a3b8;">Scanning 12 assets for Ichimoku Kumo breakouts clearing 70% VAH with TD Setup &lt; 9. (Longs paused in Bear).</span>
                </div>
                <div>
                    <b style="color: #facc15;">⚡ 5m Chop Grid Maker (Dip-Buys):</b><br>
                    <span style="color: #94a3b8;">Armed for post-only maker limit dip-buys on extreme oversold wicks (RSI &lt; 30 &amp; ADX &lt; 24).</span>
                </div>
                <div>
                    <b style="color: #f43f5e;">⚡ Quick Scalp Short Engine:</b><br>
                    <span style="color: #94a3b8;">Active high-velocity scalper: Bear Relief Bounces &amp; Chop Tops (+0.90% TP / 45m Hard Cutoff / Zero Swings).</span>
                </div>
                <div>
                    <b style="color: #a855f7;">🔒 Sizing &amp; Risk Parameters:</b><br>
                    <span style="color: #94a3b8;">$85 Margin per slot | Max 4 concurrent slots | Native exchange Bitget stops.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 1.2b Watchlist Candlestick Radar (On-Demand Chart Inspection) ─────────
        st.subheader("📈 Watchlist Technical Radar & Quantitative Overlays")
        st.caption("Inspect live multi-timeframe price action, EMAs, 60% Volume Profile (POC/VAH/VAL), and TD-Sequential counts across watchlist assets.")

        rc_col1, rc_col2, rc_col3, rc_col4 = st.columns([2, 1.5, 1.5, 3])
        with rc_col1:
            radar_options = getattr(config, "SYMBOLS", [
                "BTC/USDT", "ETH/USDT", "SOL/USDT", "NEAR/USDT", "ADA/USDT", 
                "AVAX/USDT", "LINK/USDT", "BNB/USDT", "XRP/USDT", "SUI/USDT", "INJ/USDT", "APT/USDT"
            ])
            radar_sym = st.selectbox(
                "Select Watchlist Asset",
                options=radar_options,
                index=0,
                key="radar_chart_sym"
            )
        with rc_col2:
            radar_tf = st.selectbox(
                "Candle Timeframe",
                options=["5m", "15m", "1h", "4h"],
                index=3,
                key="radar_chart_tf"
            )
        with rc_col3:
            radar_limit = st.selectbox(
                "Lookback Bars",
                options=[30, 50, 100],
                index=1,
                key="radar_chart_limit"
            )
        with rc_col4:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            r_show_emas = st.checkbox("Show Trend EMAs (9 & 21)", value=True, key="radar_show_emas")
            r_show_vp = st.checkbox("Show Volume Profile (POC/VAH/VAL)", value=True, key="radar_show_vp")

        radar_clean = radar_sym.replace("/", "").replace("-", "")
        r_live_p = num(live_prices.get(radar_clean), 0.0)

        # 🔬 Live Microstructure & Alpha Telemetry Banner for Selected Asset
        # A symbol entry that lacks a key used to publish 0.00 for it and a green verdict
        # built on that zero. Each field is now measured or absent on its own; an absent one
        # reads "not measured" in the banner's own monospace, and its label reads NOT
        # MEASURED in neutral grey.
        try:
            if os.path.exists(MICROSTRUCTURE_STATE_FILE):
                with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f_radar_ms:
                    r_ms_data = json.load(f_radar_ms)
                r_sym_code = radar_sym.replace("/", "").replace("-", "").upper()
                r_sym_info = r_ms_data.get("symbols", {}).get(r_sym_code, {})
                if r_sym_info:
                    r_absent = f"no {radar_sym} microstructure telemetry published"
                    r_mlofi_st = _published_number(r_sym_info.get("mlofi"), r_absent)
                    r_micro_price_st = _published_number(r_sym_info.get("micro_price"), r_absent)
                    r_micro_bps_st = _published_number(r_sym_info.get("micro_price_bps"), r_absent)
                    r_vpin_st = _published_number(r_sym_info.get("vpin"), r_absent)
                    r_hawkes_spoof_st = _published_number(r_sym_info.get("hawkes_spoof_index"), r_absent)
                    r_hawkes_rho_st = _published_number(r_sym_info.get("hawkes_rho"), r_absent)
                    r_hawkes_veto_st = _published_boolean(r_sym_info.get("hawkes_veto"), r_absent)
                    r_kyle_lambda_st = _published_number(r_sym_info.get("kyle_lambda"), r_absent)
                    r_kyle_pctile_st = _published_number(r_sym_info.get("kyle_lambda_pctile"), r_absent)
                    r_lvr_tax_st = _published_number(r_sym_info.get("lvr_tax_bps"), r_absent)
                    r_tox_halt_st = _published_boolean(r_sym_info.get("toxicity_halt"), r_absent)

                    r_mlofi_text = f"{r_mlofi_st['value']:+.2f}" if r_mlofi_st["measured"] else "not measured"
                    r_m_col = ("#00ff66" if r_mlofi_st["value"] > 0.05 else ("#ff003c" if r_mlofi_st["value"] < -0.05 else "#94a3b8")) if r_mlofi_st["measured"] else "#94a3b8"
                    r_micro_price_text = f"${r_micro_price_st['value']:,.4f}" if r_micro_price_st["measured"] else "not measured"
                    r_micro_price_color = "#00e5ff" if r_micro_price_st["measured"] else "#94a3b8"
                    r_micro_bps_text = f"{r_micro_bps_st['value']:+.2f} bps" if r_micro_bps_st["measured"] else "not measured"
                    r_bps_color = ("#00ff66" if r_micro_bps_st["value"] >= 0 else "#ff003c") if r_micro_bps_st["measured"] else "#94a3b8"
                    r_vpin_text = f"{r_vpin_st['value']:.3f}" if r_vpin_st["measured"] else "not measured"
                    r_v_col = ("#00ff66" if r_vpin_st["value"] < 0.45 else ("#f59e0b" if r_vpin_st["value"] < 0.70 else "#ff003c")) if r_vpin_st["measured"] else "#94a3b8"
                    r_v_lbl = ("Benign Flow" if r_vpin_st["value"] < 0.45 else ("Caution Flow" if r_vpin_st["value"] < 0.70 else "TOXIC CASCADE")) if r_vpin_st["measured"] else "NOT MEASURED"
                    r_kyle_lambda_text = f"{r_kyle_lambda_st['value']:.4f}" if r_kyle_lambda_st["measured"] else "not measured"
                    r_kyle_pctile_text = f"{r_kyle_pctile_st['value']*100:.0f}%" if r_kyle_pctile_st["measured"] else "not measured"
                    r_kyle_col = ("#00ff66" if r_kyle_pctile_st["value"] < 0.60 else ("#f59e0b" if r_kyle_pctile_st["value"] < 0.85 else "#ff003c")) if r_kyle_pctile_st["measured"] else "#94a3b8"
                    r_lvr_tax_text = f"{r_lvr_tax_st['value']:.2f} bps" if r_lvr_tax_st["measured"] else "not measured"
                    r_lvr_color = ("#ff003c" if r_lvr_tax_st["value"] >= 10.0 else "#38bdf8") if r_lvr_tax_st["measured"] else "#94a3b8"
                    # Each Hawkes field publishes its own reading: a missing `hawkes_veto` must not
                    # make the banner disown a spoof index or a ρ it actually measured. Only the
                    # combined verdict below (colour + label) may be bundled, because it genuinely
                    # needs all three keys. Note the two formats differ and are not interchangeable.
                    r_h_spoof_text = f"{r_hawkes_spoof_st['value']:.3f}" if r_hawkes_spoof_st["measured"] else "not measured"
                    r_h_rho_text = f"{r_hawkes_rho_st['value']:.2f}" if r_hawkes_rho_st["measured"] else "not measured"
                    if r_hawkes_veto_st["measured"] and r_hawkes_spoof_st["measured"] and r_hawkes_rho_st["measured"]:
                        r_h_col = "#ff003c" if (r_hawkes_veto_st["value"] or r_hawkes_spoof_st["value"] > 0.72 or r_hawkes_rho_st["value"] >= 1.0) else ("#f59e0b" if r_hawkes_spoof_st["value"] > 0.45 else "#00ff66")
                        r_h_lbl = "VETO" if r_hawkes_veto_st["value"] else ("ELEVATED" if r_hawkes_spoof_st["value"] > 0.45 else "STATIONARY")
                    else:
                        r_h_col = "#94a3b8"
                        r_h_lbl = "NOT MEASURED"
                    if r_tox_halt_st["measured"]:
                        r_lvr_badge = "<b style='color: #ff003c;'>[TOXIC HALT]</b>" if r_tox_halt_st["value"] else ""
                    else:
                        r_lvr_badge = "<b style='color: #94a3b8;'>[NOT MEASURED]</b>"

                    st.markdown(f"""
                    <div style="background: rgba(13, 18, 31, 0.85); border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 10px 16px; margin-top: 10px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                        <span>🔬 <b>{radar_sym} Alpha Telemetry:</b></span>
                        <span>Micro-Price: <b style="color: {r_micro_price_color};">{r_micro_price_text}</b> (<b style="color: {r_bps_color};">{r_micro_bps_text}</b>)</span>
                        <span>MLOFI (10-Level): <b style="color: {r_m_col};">{r_mlofi_text}</b></span>
                        <span>VPIN: <b style="color: {r_v_col};">{r_vpin_text} [{r_v_lbl}]</b></span>
                        <span>Hawkes Spoof: <b style="color: {r_h_col};">{r_h_spoof_text} (ρ={r_h_rho_text}) [{r_h_lbl}]</b></span>
                        <span>Kyle's λ (Impact): <b style="color: {r_kyle_col};">{r_kyle_lambda_text} ({r_kyle_pctile_text})</b></span>
                        <span>LVR Tax (σ²/8): <b style="color: {r_lvr_color};">{r_lvr_tax_text}</b> {r_lvr_badge}</span>
                        <span style="color: #94a3b8; font-size: 0.75rem;">🛡️ CRC32 Verified</span>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass
        
        fig_radar = build_position_candlestick_chart(
            symbol=radar_sym,
            side="long",
            entry_p=0.0,
            sl=0.0,
            hard_sl=0.0,
            tp=0.0,
            curr_p=r_live_p,
            est_liq=0.0,
            timeframe=radar_tf,
            limit=radar_limit,
            show_emas=r_show_emas,
            show_vp=r_show_vp
        )
        if fig_radar:
            st.plotly_chart(fig_radar, width="stretch", key=f"fig_radar_{radar_clean}_{radar_tf}")
        else:
            st.info(f"Connecting to live feed for {radar_sym}... Select timeframe or refresh.")

    # ── Sovereign Bitcoin Anchor & Multi-Asset Relative Strength Cards ───────
    st.markdown("---")
    
    # Fetch BTC Anchor Telemetry
    btc_anchor_state = query_system_state("btc_anchor_state", "HEALTHY_EXPANSION")
    is_alt_paused = state_flag("ALT_LONG_PAUSE")
    try:
        btc_risk_score = state_num("btc_contagion_risk_score", 20.0)
    except Exception:
        btc_risk_score = 20.0
    btc_directive = query_system_state("btc_anchor_directive", "Bitcoin in healthy posture; normal fleet operations active.")
    try:
        btc_pivot = state_num("btc_key_pivot", 84500.0)
        btc_pivot_str = f"${btc_pivot:,.0f}"
    except Exception:
        btc_pivot_str = "$84,500"

    # Fetch Relative Strength Matrix Telemetry
    raw_leaders = query_system_state("ai_asset_leaders", "NEAR/USDT, AVAX/USDT, XRP/USDT")
    raw_avoid = query_system_state("ai_asset_avoid", "BTC/USDT, ETH/USDT, BNB/USDT, DOGE/USDT")
    rs_rankings_raw = query_system_state("ai_asset_rankings", "")
    rs_rationale = ""
    if rs_rankings_raw:
        try:
            rs_obj = json.loads(rs_rankings_raw)
            rs_rationale = rs_obj.get("fleet_focus_rationale", "")
        except Exception:
            pass

    col_anchor, col_matrix = st.columns(2)
    with col_anchor:
        anchor_color = "#00ff66" if not is_alt_paused else "#ff003c"
        anchor_badge = "🟢 NORMAL FLEET ACCESS" if not is_alt_paused else "⏸️ ALT LONGS PAUSED"
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {anchor_color}; border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <b style="color: #ffffff; font-size: 0.95rem;">⚓ The Bitcoin Anchor (Cross-Market Contagion Filter)</b>
                <span style="background: {'rgba(0,255,102,0.15)' if not is_alt_paused else 'rgba(255,0,60,0.15)'}; color: {anchor_color}; border: 1px solid {anchor_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75rem;">
                    {anchor_badge}
                </span>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.5; margin-bottom: 6px;">
                BTC State: <b style="color: #00e5ff;">{btc_anchor_state}</b> | Contagion Risk: <b style="color: {'#00ff66' if btc_risk_score < 40 else ('#ff9900' if btc_risk_score < 70 else '#ff003c')};">{btc_risk_score:.0f}/100</b> | Pivot: <b style="color: #e2e8f0;">{btc_pivot_str}</b>
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; background: rgba(255,255,255,0.02); padding: 8px 10px; border-radius: 4px; border-left: 2px solid rgba(255,255,255,0.1);">
                <i>"{btc_directive}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_matrix:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 14px 18px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <b style="color: #ffffff; font-size: 0.95rem;">👑 Multi-Asset Relative Strength & Leader Gate</b>
                <span style="background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid #00e5ff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.75rem;">
                    30M ROTATION
                </span>
            </div>
            <div style="font-size: 0.8rem; line-height: 1.6; margin-bottom: 6px;">
                <span style="color: #64748b;">Alpha Leaders:</span> <b style="color: #00ff66;">{raw_leaders}</b><br>
                <span style="color: #64748b;">Avoid / Chop:</span> <b style="color: #ff003c; text-decoration: line-through;">{raw_avoid}</b>
            </div>
            <div style="font-size: 0.78rem; color: #cbd5e1; background: rgba(255,255,255,0.02); padding: 8px 10px; border-radius: 4px; border-left: 2px solid rgba(255,255,255,0.1);">
                <i>"{rs_rationale if rs_rationale else 'Capital strictly deployed to relative strength leaders with positive order flow.'}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📜 Relative Strength & Leader Rotation History (Audit Log)", expanded=False):
        df_rs_hist = query_db(TRADING_LEDGER_DB, "SELECT timestamp, priority_leaders, secondary_tier, avoid_chop_list, rationale FROM relative_strength_history ORDER BY id DESC LIMIT 10")
        if not df_rs_hist.empty:
            df_rs_hist['Time (SAST)'] = df_rs_hist['timestamp'].apply(utc_to_sast)
            show_rs = df_rs_hist[['Time (SAST)', 'priority_leaders', 'secondary_tier', 'avoid_chop_list', 'rationale']].rename(
                columns={
                    'priority_leaders': 'Alpha Leaders',
                    'secondary_tier': 'Secondary',
                    'avoid_chop_list': 'Avoid List',
                    'rationale': 'AI Rationale'
                }
            )
            st.dataframe(show_rs, width="stretch", hide_index=True)
        else:
            st.caption("Relative strength audit history synchronizing...")

    st.subheader("📡 Top 10 Watchlist Radar (Live Technical Screening)")
    
    raw_allowed = query_system_state("ALLOWED_SYMBOLS")
    if raw_allowed:
        try:
            top_coins = json.loads(raw_allowed)
        except Exception:
            top_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]
    else:
        top_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]

    leaders_list = [c.strip().upper() for c in raw_leaders.split(",") if c.strip()]
    avoid_list = [c.strip().upper() for c in raw_avoid.split(",") if c.strip()]

    radar_data = []
    for coin in top_coins:
        coin_clean = coin.replace("/", "").replace("-", "")
        is_active = any(coin_clean in c for c in active_coins_set)
        c_price = live_prices.get(coin_clean, 0.0)
        is_core = coin in ["BTC/USDT", "ETH/USDT"]
        coin_lev = config.get_asset_leverage(coin) if hasattr(config, "get_asset_leverage") else (10 if "BTC" in coin else 5)

        # AI Fleet Tier designation
        if any(coin.upper() == l or coin_clean == l.replace("/", "") for l in leaders_list):
            ai_tier = "⭐ ALPHA LEADER"
        elif any(coin.upper() == a or coin_clean == a.replace("/", "") for a in avoid_list):
            ai_tier = "🚫 AVOID / CHOP"
        else:
            ai_tier = "⏳ SECONDARY"
        
        radar_data.append({
            "Coin": coin,
            "Live Price": f"${c_price:,.4f}" if c_price > 0 else "---",
            "AI Fleet Tier": ai_tier,
            "Role": "⭐ Core (Reserved Slot)" if is_core else "Altcoin (Dynamic Slot)",
            "Dynamic Lev": f"{coin_lev}x Isolated",
            "Fleet Status": "🟢 ACTIVE POSITION" if is_active else "⏳ SCANNING (RADAR)",
            "Macro Direction": {"BULLISH": "LONG ONLY", "BEARISH": "SHORT ONLY"}.get(macro_trend, "NO STANCE")
        })
    st.dataframe(pd.DataFrame(radar_data), width="stretch", hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: CLOSED TRADES & ANALYTICS (Filters, CSV Export & Chronological Curve)
# ═════════════════════════════════════════════════════════════════════════════
with tab_ledger:
    st.subheader(f"📜 Historical Closed Trades Ledger ({total_closed_count} Completed Trades)")

    # Capital Baseline & Return Summary Banner
    st.markdown(f"""
    <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 4px solid #00e5ff; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Total Deposited Capital</span><br>
                <span style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{capital['deposited_usdt']}</span>
                <span style="color: #64748b; font-size: 0.8rem;"> ({capital['tranches']})</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Current Portfolio Value</span><br>
                <span style="color: #38bdf8; font-size: 1.15rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{bal_display} USDT</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Real Net Profit</span><br>
                <span style="color: {profit_color}; font-size: 1.15rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">
                    {_profit_display(total_profit, total_roi_pct)}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not df_all_trades.empty:
        # P2: Ledger Filters Row
        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        with fcol1:
            sym_list = ["ALL ASSETS"] + sorted([str(s) for s in df_all_trades['symbol'].dropna().unique() if s])
            f_symbol = st.selectbox("Filter by Symbol", sym_list, index=0)
        with fcol2:
            f_outcome = st.selectbox("Filter by Outcome", ["ALL OUTCOMES", "Wins Only (PnL > 0)", "Losses Only (PnL <= 0)"], index=0)
        with fcol3:
            f_sort = st.selectbox("Sort Order", ["Newest First", "Oldest First", "Highest Profit", "Largest Loss"], index=0)

        df_filtered = df_all_trades.copy()
        if f_symbol != "ALL ASSETS":
            df_filtered = df_filtered[df_filtered['symbol'] == f_symbol]
        if f_outcome == "Wins Only (PnL > 0)":
            df_filtered = df_filtered[df_filtered['pnl'].astype(float) > 0]
        elif f_outcome == "Losses Only (PnL <= 0)":
            df_filtered = df_filtered[df_filtered['pnl'].astype(float) <= 0]

        if f_sort == "Newest First":
            df_filtered = df_filtered.sort_values(by='dt_sort', ascending=False)
        elif f_sort == "Oldest First":
            df_filtered = df_filtered.sort_values(by='dt_sort', ascending=True)
        elif f_sort == "Highest Profit":
            df_filtered = df_filtered.sort_values(by='pnl', ascending=False)
        elif f_sort == "Largest Loss":
            df_filtered = df_filtered.sort_values(by='pnl', ascending=True)

        # P2: CSV Export Button
        csv_bytes = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Export Filtered Ledger to CSV ({len(df_filtered)} Trades)",
            data=csv_bytes,
            file_name=f"sovereign_swarm_trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download filtered trade history in standardized CSV format for quantitative auditing."
        )

        # Format ledger for display
        df_ledger_display = df_filtered.copy()
        df_ledger_display['Timestamp (SAST)'] = df_ledger_display['timestamp'].apply(utc_to_sast)
        df_ledger_display['PnL ($)'] = df_ledger_display['pnl'].apply(lambda x: f"{'+$' if num(x)>=0 else '-$'}{abs(num(x)):.4f}")
        df_ledger_display['Result'] = df_ledger_display['pnl'].apply(lambda x: "🟢 WIN" if num(x)>0 else "🔴 LOSS")
        df_ledger_display['Entry Price'] = df_ledger_display['entry_price'].apply(lambda x: f"${num(x):,.4f}" if num(x) else "$0.00")
        df_ledger_display['Exit Price'] = df_ledger_display['exit_price'].apply(lambda x: f"${num(x):,.4f}" if num(x) else "$0.00")
        df_ledger_display['Notes'] = df_ledger_display['notes'].fillna("Completed")
        
        cols_to_show = ['Timestamp (SAST)', 'symbol', 'side', 'Entry Price', 'Exit Price', 'qty', 'PnL ($)', 'Result', 'Notes']
        valid_cols = [c for c in cols_to_show if c in df_ledger_display.columns]
        st.dataframe(df_ledger_display[valid_cols].rename(columns={'symbol': 'Asset', 'side': 'Side', 'qty': 'Size'}), width="stretch", hide_index=True)
        
        # ── P1: Chronological PnL Equity Curve & Drawdown Chart ───────────────
        st.markdown("---")
        st.subheader("📈 Chronological Cumulative Realized PnL & Drawdown Curve")
        
        df_chart = df_all_trades.copy()
        df_chart['pnl_num'] = pd.to_numeric(df_chart['pnl'], errors='coerce').fillna(0.0)
        df_chart['dt'] = pd.to_datetime(df_chart['timestamp'], utc=True, errors='coerce')
        df_chart = df_chart.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)
        
        if not df_chart.empty:
            df_chart['time_sast'] = df_chart['dt'].apply(utc_to_sast)
            df_chart['cum_pnl'] = df_chart['pnl_num'].cumsum()
            df_chart['hwm'] = df_chart['cum_pnl'].cummax()
            df_chart['drawdown'] = df_chart['cum_pnl'] - df_chart['hwm']
            
            fig_equity = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=("Cumulative Realized PnL (USDT)", "Realized Equity Drawdown (USDT)"),
                row_heights=[0.75, 0.25]
            )
            
            # Cumulative PnL line
            fig_equity.add_trace(go.Scatter(
                x=df_chart['time_sast'],
                y=df_chart['cum_pnl'],
                mode='lines+markers',
                name='Cumulative PnL',
                line=dict(color='#00ff66', width=2.5),
                marker=dict(size=4, color='#00e5ff'),
                fill='tozeroy',
                fillcolor='rgba(0, 255, 102, 0.08)',
                hovertemplate="%{x}<br>Cumulative PnL: $%{y:+.4f} USDT<extra></extra>"
            ), row=1, col=1)
            
            # High Water Mark
            fig_equity.add_trace(go.Scatter(
                x=df_chart['time_sast'],
                y=df_chart['hwm'],
                mode='lines',
                name='High Water Mark',
                line=dict(color='#00e5ff', width=1.5, dash='dot'),
                hovertemplate="%{x}<br>Peak PnL: $%{y:+.4f} USDT<extra></extra>"
            ), row=1, col=1)
            
            # Drawdown chart
            fig_equity.add_trace(go.Scatter(
                x=df_chart['time_sast'],
                y=df_chart['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff003c', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 60, 0.15)',
                hovertemplate="%{x}<br>Drawdown: $%{y:+.4f} USDT<extra></extra>"
            ), row=2, col=1)
            
            fig_equity.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13, 18, 31, 0.8)',
                margin=dict(l=20, r=20, t=40, b=20),
                height=450,
                showlegend=True,
                xaxis2=dict(title="Execution Timestamp (SAST)", tickangle=-30)
            )
            st.plotly_chart(fig_equity, width="stretch")
        
        # ── 5.3 Win Rate & Profitability by Coin Breakdown ────────────────────
        st.markdown("---")
        st.subheader("🪙 Performance & Win Rate Breakdown by Coin")
        
        coin_grp = df_all_trades.groupby('symbol').agg(
            trades=('pnl', 'count'),
            wins=('pnl', lambda x: (pd.to_numeric(x) > 0).sum()),
            net_pnl=('pnl', lambda x: pd.to_numeric(x).sum()),
            avg_pnl=('pnl', lambda x: pd.to_numeric(x).mean())
        ).reset_index()
        
        coin_grp['win_rate'] = (coin_grp['wins'] / coin_grp['trades']) * 100.0
        coin_grp = coin_grp.sort_values(by='net_pnl', ascending=False)
        
        c_col1, c_col2 = st.columns([1.2, 0.8])
        with c_col1:
            fig_coin = go.Figure()
            fig_coin.add_trace(go.Bar(
                x=coin_grp['symbol'],
                y=coin_grp['net_pnl'],
                name='Net Realized PnL ($)',
                marker=dict(
                    color=coin_grp['net_pnl'].apply(lambda x: '#00ff66' if x >= 0 else '#ff003c')
                ),
                hovertemplate="%{x}<br>Net PnL: $%{y:+.2f} USDT<extra></extra>"
            ))
            fig_coin.update_layout(
                title="Net PnL ($) per Asset",
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(13, 18, 31, 0.8)',
                yaxis_title="Realized PnL (USDT)",
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )
            st.plotly_chart(fig_coin, width="stretch")
            
        with c_col2:
            st.markdown("#### 🏆 Asset Leaderboard")
            show_coin_df = coin_grp.copy()
            show_coin_df['Win Rate (%)'] = show_coin_df['win_rate'].apply(lambda x: f"{num(x):.1f}%")
            show_coin_df['Net PnL ($)'] = show_coin_df['net_pnl'].apply(lambda x: f"{'+$' if num(x)>=0 else '-$'}{abs(num(x)):.2f}")
            show_coin_df['Avg / Trade'] = show_coin_df['avg_pnl'].apply(lambda x: f"{'+$' if num(x)>=0 else '-$'}{abs(num(x)):.2f}")
            show_coin_df = show_coin_df[['symbol', 'trades', 'Win Rate (%)', 'Net PnL ($)', 'Avg / Trade']].rename(columns={'symbol': 'Asset', 'trades': 'Trades'})
            st.dataframe(show_coin_df, width="stretch", hide_index=True)

        # ── 5.4 Trade Duration Analysis ───────────────────────────────────────
        st.markdown("---")
        st.subheader("⏱️ Trade Duration & Holding Time Analysis")
        
        if 'duration_mins' in df_all_trades.columns:
            dur_col = pd.to_numeric(df_all_trades['duration_mins'], errors="coerce")
            pnl_col = pd.to_numeric(df_all_trades['pnl'], errors="coerce") if 'pnl' in df_all_trades.columns else pd.Series(0.0, index=df_all_trades.index)
            valid_dur = df_all_trades[dur_col.notnull() & (dur_col > 0)].copy()
            valid_dur["_dur_m"] = dur_col[valid_dur.index]
            valid_dur["_pnl_n"] = pnl_col[valid_dur.index]
        else:
            valid_dur = pd.DataFrame()
        if not valid_dur.empty:
            avg_all_m = num(valid_dur["_dur_m"].mean())
            avg_win_m = num(valid_dur.loc[valid_dur["_pnl_n"] > 0, "_dur_m"].mean(), avg_all_m)
            avg_loss_m = num(valid_dur.loc[valid_dur["_pnl_n"] <= 0, "_dur_m"].mean(), avg_all_m)
            hold_ratio = (avg_win_m / avg_loss_m) if avg_loss_m > 0 else 1.0
            
            def _fmt_mins(m):
                if m >= 60.0:
                    return f"{m/60.0:.1f} hrs"
                return f"{m:.1f} mins"
            
            dur1, dur2, dur3, dur4 = st.columns(4)
            with dur1:
                st.metric("Avg Fleet Duration", _fmt_mins(avg_all_m), delta=f"{len(valid_dur)} Samples")
            with dur2:
                st.metric("Avg Winner Duration", _fmt_mins(avg_win_m), delta=f"+{max(0.0, avg_win_m - avg_loss_m):.1f}m vs losers" if avg_win_m >= avg_loss_m else "Shorter than losers")
            with dur3:
                st.metric("Avg Loser Duration", _fmt_mins(avg_loss_m), delta="Cut on SL / Invalidation")
            with dur4:
                st.metric("Hold Ratio (Win/Loss)", f"{hold_ratio:.2f}x", delta="Asymmetry (>1.0x)" if hold_ratio >= 1.0 else "Inverted (<1.0x)")
                
            st.info(f"💡 **Holding Asymmetry Insight:** Realized holding ratio is **{hold_ratio:.2f}x** across **{len(valid_dur)}** recorded trades. Winners average {_fmt_mins(avg_win_m)} while losers exit at {_fmt_mins(avg_loss_m)}.")
        else:
            st.caption("Duration metrics synchronizing with closed trades...")
    else:
        st.info("No closed trades recorded in the database yet.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: AI BOARDROOM & MEMORY
# ═════════════════════════════════════════════════════════════════════════════
with tab_ai:
    # ── 3.0 AI Model Credit, Token & Quota Telemetry ──────────────────────────
    ai_stats = fetch_ai_usage_stats()
    st.subheader("⚡ Real-Time AI Model Credit & Token Telemetry (Daily Live Monitor)")

    ai_m1, ai_m2, ai_m3, ai_m4 = st.columns(4)
    with ai_m1:
        st.metric(
            "Primary AI (Gemini)",
            f"{ai_stats['gemini_calls']} Calls",
            delta=f"{ai_stats['free_tier_remaining']} free calls left today"
        )
    with ai_m2:
        st.metric(
            "Estimated Tokens Consumed",
            f"{ai_stats['total_tokens']:,}",
            delta=f"{ai_stats['est_input_tokens']:,} in / {ai_stats['est_output_tokens']:,} out"
        )
    with ai_m3:
        st.metric(
            "Today's AI Cost",
            "$0.00",
            delta="100% Free Tier (0.0% spent)"
        )
    with ai_m4:
        st.metric(
            "Groq Failover Pool",
            f"{ai_stats['groq_pool_size']} Keys Ready",
            delta=f"{ai_stats['groq_calls']} failovers used today"
        )

    quota_pct = min(100.0, (ai_stats['gemini_calls'] / ai_stats['free_tier_limit']) * 100.0)
    quota_bar_color = "#00ff66" if quota_pct < 60 else ("#ff9900" if quota_pct < 85 else "#ff003c")

    st.markdown(f"""
    <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 18px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 6px;">
            <span style="color: #94a3b8; font-weight: 600;">
                Google Gemini Free Daily Quota Utilization (1,500 Requests/Day Allowance)
            </span>
            <span style="color: #00e5ff; font-family: 'JetBrains Mono', monospace; font-weight: bold;">
                {ai_stats['gemini_calls']} / {ai_stats['free_tier_limit']} calls ({quota_pct:.1f}%)
            </span>
        </div>
        <div style="width: 100%; background: #1e293b; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 8px;">
            <div style="width: {max(2.0, quota_pct):.1f}%; background: {quota_bar_color}; height: 100%; border-radius: 4px; transition: width 0.4s ease-in-out;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 0.74rem; color: #64748b; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px;">
            <span>🤖 <b>Macro Sentinel:</b> <span style="color: #cbd5e1;">{ai_stats['macro_calls']} continuous syntheses</span></span>
            <span>🏛️ <b>Boardroom Debates:</b> <span style="color: #cbd5e1;">{ai_stats['boardroom_calls']} candidate evaluations</span></span>
            <span>🛡️ <b>Groq Rate Limits:</b> <span style="color: #00ff66;">{ai_stats['groq_429s']} (0 throttles)</span></span>
            <span>⚡ <b>Active Models:</b> <span style="color: #00e5ff;">{ai_stats['primary_model']} (Primary)</span> / <span style="color: #94a3b8;">{ai_stats['failover_model']} (Backup)</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3.0b AI 6-Hour Strategic Macro Blueprint & Key Level Roadmap ──────────
    st.markdown("---")
    st.subheader("🗺️ AI Strategic Blueprint & Key Level Roadmap (6-Hour Institutional Cadence)")
    raw_bp = query_system_state("ai_daily_blueprint", "")
    if raw_bp:
        try:
            bp_obj = json.loads(raw_bp)
            bp_date = bp_obj.get("blueprint_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
            bp_session = bp_obj.get("session_name", "Global Macro Session")
            bp_bias = bp_obj.get("macro_bias", "DEFENSIVE_CHOP")
            bp_narrative = bp_obj.get("daily_narrative", "Session macro posture active.")
            bp_conf = num(bp_obj.get("confidence"), 0.85) * 100.0
            bp_updated = utc_to_sast(bp_obj.get("updated_at"))

            # Calculate remaining minutes in current 6h session block
            now_u = datetime.now(timezone.utc)
            next_h = ((now_u.hour // 6) + 1) * 6
            mins_left = (next_h - now_u.hour) * 60 - now_u.minute
            if next_h >= 24:
                next_refresh_str = f"00:00 UTC (~{mins_left}m remaining)"
            else:
                next_refresh_str = f"{next_h:02d}:00 UTC (~{mins_left}m remaining)"

            bias_color = "#00ff66" if "BULL" in bp_bias else ("#ff003c" if "BEAR" in bp_bias else "#ffaa00")

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(13, 18, 31, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%); 
                        border: 1px solid #1e293b; border-left: 4px solid {bias_color}; border-radius: 8px; 
                        padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
                            📅 Session: {bp_date} • {bp_session}
                        </span>
                        <span style="background: rgba({ '0,255,102' if 'BULL' in bp_bias else ('255,0,60' if 'BEAR' in bp_bias else '255,170,0') }, 0.15); color: {bias_color}; border: 1px solid {bias_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.78rem; margin-left: 10px;">
                            🧭 {bp_bias}
                        </span>
                    </div>
                    <span style="font-size: 0.75rem; color: #64748b; font-family: monospace;">
                        Updated: {bp_updated} | Next 6h Refresh: {next_refresh_str} (Conf: {bp_conf:.0f}%)
                    </span>
                </div>
                <div style="font-size: 0.90rem; color: #e2e8f0; line-height: 1.5; margin-top: 6px;">
                    {bp_narrative}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Asset Roadmap Table
            roadmaps = bp_obj.get("asset_roadmaps", {})
            if roadmaps:
                bp_rows = []
                for a_sym, a_data in roadmaps.items():
                    if isinstance(a_data, dict):
                        bp_rows.append({
                            "Asset": a_sym,
                            "Bias": a_data.get("bias", "RANGE"),
                            "Key Support": f"${num(a_data.get('key_support'), 0.0):,.2f}" if num(a_data.get('key_support'), 0.0) > 0 else "---",
                            "Key Resistance": f"${num(a_data.get('key_resistance'), 0.0):,.2f}" if num(a_data.get('key_resistance'), 0.0) > 0 else "---",
                            "Invalidation": f"${num(a_data.get('invalidation_level'), 0.0):,.2f}" if num(a_data.get('invalidation_level'), 0.0) > 0 else "---",
                            "Tactical Playbook": a_data.get("tactical_plan", "Standard risk limits active.")
                        })
                if bp_rows:
                    st.dataframe(pd.DataFrame(bp_rows), width="stretch", hide_index=True)
        except Exception as bp_ui_err:
            st.info(f"Session blueprint synchronizing: {bp_ui_err}")
    else:
        st.info("AI Strategic Blueprint scheduled to generate on first session cycle.")

    st.markdown("---")
    st.subheader("⚔️ Live Adversarial Red Team Deliberation Chamber")
    
    debate_raw = query_system_state("latest_ai_debate", "")
    if debate_raw:
        try:
            d_obj = json.loads(debate_raw)
            d_sym = d_obj.get("symbol", "BTC/USDT")
            d_side = d_obj.get("side", "SHORT")
            d_exec = d_obj.get("execute", False)
            d_score = num(d_obj.get("conviction_score"), 0.0) * 100.0
            d_red = d_obj.get("red_team_objection", "No active objections recorded.")
            d_blue = d_obj.get("alpha_thesis", "Momentum continuation confirmed.")
            d_rat = d_obj.get("final_rationale", "Adversarial deliberation rendered.")
            d_time = utc_to_sast(d_obj.get("timestamp"))

            d_badge_color = "#00ff66" if d_exec else "#ff003c"
            d_status_text = f"APPROVED ({d_score:.0f}% Conviction)" if d_exec else f"VETOED ({d_score:.0f}% Conviction)"

            st.markdown(f"""
            <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #1e293b; padding-bottom: 8px;">
                    <div>
                        <b style="font-size: 1.05rem; color: #ffffff;">🪙 Candidate Setup: {d_sym} ({d_side})</b>
                        <span style="color: #64748b; font-size: 0.75rem; margin-left: 12px;">Deliberation at {d_time}</span>
                    </div>
                    <div>
                        <span style="background: {'rgba(0,255,102,0.15)' if d_exec else 'rgba(255,0,60,0.15)'}; color: {d_badge_color}; border: 1px solid {d_badge_color}; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">
                            ⚖️ ARBITER: {d_status_text}
                        </span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
                    <div style="background: rgba(255, 0, 60, 0.06); border-left: 3px solid #ff003c; padding: 10px 14px; border-radius: 4px;">
                        <b style="color: #ff003c; font-size: 0.85rem;">🛡️ Bearish Red Team (Risk Officer)</b>
                        <p style="color: #e2e8f0; font-size: 0.82rem; margin: 6px 0 0 0; line-height: 1.4;">{d_red}</p>
                    </div>
                    <div style="background: rgba(0, 229, 255, 0.06); border-left: 3px solid #00e5ff; padding: 10px 14px; border-radius: 4px;">
                        <b style="color: #00e5ff; font-size: 0.85rem;">🚀 Bullish Alpha Hunter (Tactical Officer)</b>
                        <p style="color: #e2e8f0; font-size: 0.82rem; margin: 6px 0 0 0; line-height: 1.4;">{d_blue}</p>
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 8px 12px; background: rgba(0,0,0,0.25); border-radius: 4px; font-size: 0.8rem; color: #94a3b8;">
                    🏛️ <b>Supreme Arbiter Ruling:</b> <span style="color: #ffffff;">{d_rat}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as deb_err:
            st.info(f"Red team deliberation chamber synchronizing: {deb_err}")
    else:
        st.info("Deliberation chamber awaiting next candidate signal...")

    # ── 3.0c AI Cognitive Regime Arbiter & Breakout Trap Gate Shield ────────
    st.markdown("---")
    st.subheader("⚖️ AI Cognitive Regime Arbiter & Breakout Trap Gate Shield")

    arb_bias_state = _published_state(query_system_state("ai_regime_bias", ""), "arbiter published no regime bias")
    arb_risk_num = _optional_float(query_system_state("ai_regime_trap_risk", ""))
    arb_thesis_state = _published_state(query_system_state("ai_regime_thesis", ""), "arbiter published no thesis")
    arb_lev_state = _published_state(query_system_state("recommended_leverage", ""), "no recommended leverage published")

    col_arb1, col_arb2, col_arb3 = st.columns([1.2, 1, 2.0])
    with col_arb1:
        if arb_bias_state["measured"]:
            arb_lev_delta = (f"Dynamic Cap: {html.escape(arb_lev_state['text'])}x"
                             if arb_lev_state["measured"] else "Dynamic Cap: not measured")
            st.metric("Cognitive Regime Bias", html.escape(arb_bias_state["text"]),
                      delta=arb_lev_delta, delta_color="normal")
        else:
            st.metric("Cognitive Regime Bias", "not measured",
                      delta=arb_bias_state["reason"], delta_color="off")
    with col_arb2:
        if arb_risk_num is None:
            st.metric("Institutional Trap Risk", "not measured",
                      delta="trap risk was not published", delta_color="off")
        else:
            st.metric("Institutional Trap Risk", f"{arb_risk_num * 100:.0f}%",
                      delta="Safe Expansion" if arb_risk_num < 0.5 else "High Trap Risk",
                      delta_color="normal" if arb_risk_num < 0.5 else "inverse")
    with col_arb3:
        thesis_text = (html.escape(arb_thesis_state["text"]) if arb_thesis_state["measured"]
                       else "not measured")
        thesis_note = "" if arb_thesis_state["measured"] else _absence_note(arb_thesis_state["reason"])
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #cbd5e1;">
            <b style="color: #00e5ff;">🧠 Arbiter Structural Thesis:</b><br/>
            {thesis_text}{thesis_note}
        </div>
        """, unsafe_allow_html=True)

    # ── 3.0d Per-Asset Decoupled Regime Matrix (Alpha Leaderboard) ────────────
    raw_decoupled = query_system_state("ai_decoupled_regime_matrix", "")
    if raw_decoupled:
        try:
            decoupled_dict = json.loads(raw_decoupled)
            if decoupled_dict:
                st.markdown("<h5 style='margin-top: 15px; color: #94a3b8;'>🌐 Per-Asset Decoupled Micro-Regimes (Autonomous Divergence Matrix)</h5>", unsafe_allow_html=True)
                rows_d = []
                for sym, info in decoupled_dict.items():
                    d_reg = info.get("decoupled_regime", "BTC_BETA_SLAVE")
                    allowed = "✅ YES (Alpha Runner)" if info.get("independent_trade_allowed") else "🔒 Bound to BTC"
                    tier = info.get("tier", "SECONDARY")
                    score = info.get("edge_score", 50)
                    posture = info.get("posture", "NEUTRAL_CHOP")
                    notes = info.get("notes", "")
                    rows_d.append({
                        "Asset": sym,
                        "Decoupled Regime": d_reg,
                        "Tier": tier,
                        "Alpha Edge": f"{score}/100",
                        "Posture": posture,
                        "Independent Trade Clearance": allowed,
                        "Strategic Notes": notes
                    })
                st.dataframe(pd.DataFrame(rows_d), width="stretch", hide_index=True)
        except Exception:
            pass

    # ── 3.0d2 AI Event Catalyst & Horizon Arbiter (Breaking News & Volatility Radar) ──
    st.markdown("---")
    st.subheader("⚡ AI Event Catalyst & Horizon Arbiter (Breaking News & Volatility Radar)")
    st.caption("Autonomous institutional classification distinguishing between flash liquidity shocks (rebound traps) and secular macro regime shifts (high-conviction shorts/breakouts).")

    ev_playbook = query_system_state("ai_event_playbook", "NORMAL_OPERATIONS")
    ev_horizon = query_system_state("ai_event_horizon", "NEUTRAL")
    ev_headline = query_system_state("ai_event_headline", "No major high-urgency breaking news catalyst active.")
    ev_summary = query_system_state("ai_event_summary", "Autonomous news sentinel monitoring global feeds.")
    ev_category = query_system_state("ai_event_category", "NONE")
    ev_raw = query_system_state("ai_event_catalyst", "{}")
    ev_upd = query_system_state("ai_event_updated_at", "")

    # Playbook badge color
    pb_color = "#00e5ff"
    pb_bg = "rgba(0, 229, 255, 0.1)"
    if "SHORT" in ev_playbook or "BEAR" in ev_horizon:
        pb_color = "#ff003c"
        pb_bg = "rgba(255, 0, 60, 0.12)"
    elif "REBOUND" in ev_playbook or "FLASH" in ev_horizon:
        pb_color = "#f59e0b"
        pb_bg = "rgba(245, 158, 11, 0.12)"
    elif "BULL" in ev_playbook or "EXPANSION" in ev_playbook:
        pb_color = "#00ff66"
        pb_bg = "rgba(0, 255, 102, 0.12)"

    col_ev1, col_ev2, col_ev3 = st.columns([1.2, 1.2, 2.2])
    with col_ev1:
        st.metric("Event Horizon", ev_horizon.replace("_", " "), delta=ev_category.replace("_", " "))
    with col_ev2:
        st.markdown(f"""
        <div style="background: {pb_bg}; border: 1px solid {pb_color}; border-radius: 6px; padding: 10px 12px; text-align: center;">
            <span style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: bold;">Active Tactical Playbook</span><br/>
            <b style="color: {pb_color}; font-size: 1.05rem; font-family: monospace;">{ev_playbook}</b>
        </div>
        """, unsafe_allow_html=True)
    with col_ev3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #cbd5e1;">
            <b style="color: {pb_color};">📡 Latest Breaking Catalyst:</b><br/>
            <i>"{ev_headline}"</i><br/>
            <span style="color: #94a3b8; font-size: 0.78rem; margin-top: 4px; display: inline-block;"><b>Synthesis:</b> {ev_summary}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 3.0e Real-Time Rule Enforcement Gate & Learned Memory ────────────────
    st.markdown("---")
    st.subheader("📜 Dynamic Rule Enforcement Ledger (Closed-Loop Forensic Safeguards)")
    st.caption("The Boardroom & Red Team cross-examine every trade against these dynamically distilled institutional rules from trading_ledger.db:")
    df_rules = query_db(TRADING_LEDGER_DB, "SELECT rule_id, category, rule_text, confidence, created_at FROM learned_rules WHERE confidence >= 0.70 ORDER BY confidence DESC, rule_id DESC LIMIT 15")
    if not df_rules.empty:
        df_rules["confidence"] = df_rules["confidence"].apply(lambda x: f"{num(x):.0%}")
        df_rules["created_at"] = df_rules["created_at"].apply(utc_to_sast)
        df_rules.rename(columns={
            "rule_id": "Rule ID",
            "category": "Category",
            "rule_text": "Enforced Principle / Forensic Directive",
            "confidence": "Weight",
            "created_at": "Learned Date"
        }, inplace=True)
        st.dataframe(df_rules, width="stretch", hide_index=True)
    else:
        st.info("No learned rules at/above 0.70 confidence in the trading ledger.")

    # ── Real-Time Active Position AI Health Radar ─────────────────────────────
    st.markdown("---")
    st.subheader("🩺 Real-Time AI Position Health Monitor (Groq Low-Latency Diagnostics)")
    
    if not df_open.empty:
        health_rows = []
        for _, pos_r in df_open.iterrows():
            pos_sym = str(pos_r['symbol'])
            clean_s = pos_sym.upper().replace("/", "").replace("-", "")
            raw_h = query_system_state(f"ai_health_{clean_s}", "")
            if raw_h:
                try:
                    h_data = json.loads(raw_h)
                    health_rows.append({
                        "Asset": pos_sym,
                        "Side": str(pos_r.get("side", "LONG")).upper(),
                        "Health Grade": h_data.get("health_grade", "PRISTINE"),
                        "AI Verdict": h_data.get("verdict", "HOLD"),
                        "Confidence": f"{num(h_data.get('confidence'), 0.5)*100:.0f}%",
                        "Diagnosis": h_data.get("diagnosis", "Normal"),
                        "Suggested SL": f"${num(h_data.get('suggested_sl')):,.4f}" if num(h_data.get('suggested_sl')) else "---",
                        "Last Evaluated": utc_to_sast(h_data.get("checked_at"))
                    })
                except Exception:
                    pass
        if health_rows:
            st.dataframe(pd.DataFrame(health_rows), width="stretch", hide_index=True)
        else:
            st.info("15-minute AI health check scheduled for active positions.")
    else:
        st.info("No active positions currently requiring AI health monitoring.")

    # ── 3.1b In-Flight AI Shepherd History & Tape Decisions ───────────────────
    st.markdown("---")
    st.subheader("🦅 In-Flight AI Shepherd & Runner History (Trading Ledger)")
    try:
        df_shep = query_db(TRADING_LEDGER_DB,
            "SELECT timestamp, symbol, side, entry_price, current_price, unrealized_roe_pct, r_multiple, holding_mins, verdict, tape_condition, suggested_sl, sl_ratcheted, diagnosis FROM inflight_shepherd_logs ORDER BY id DESC LIMIT 15")
        if not df_shep.empty:
            df_shep["timestamp"] = df_shep["timestamp"].apply(utc_to_sast)
            df_shep["ROE %"] = df_shep["unrealized_roe_pct"].apply(lambda x: f"{num(x):+.2f}%")
            df_shep["R-Mult"] = df_shep["r_multiple"].apply(lambda x: f"{num(x):+.2f}R")
            df_shep["Ratcheted"] = df_shep["sl_ratcheted"].apply(lambda x: "YES" if x else "NO")
            df_shep.rename(columns={
                "timestamp": "Evaluated At",
                "symbol": "Asset",
                "side": "Side",
                "entry_price": "Entry",
                "current_price": "Mark Price",
                "holding_mins": "Hold (m)",
                "verdict": "Verdict",
                "tape_condition": "L2 Tape",
                "suggested_sl": "Suggested SL",
                "diagnosis": "AI Diagnosis"
            }, inplace=True)
            cols_order = ["Evaluated At", "Asset", "Side", "Entry", "Mark Price", "ROE %", "R-Mult", "Hold (m)", "Verdict", "L2 Tape", "Suggested SL", "Ratcheted", "AI Diagnosis"]
            st.dataframe(df_shep[[c for c in cols_order if c in df_shep.columns]], width="stretch", hide_index=True)
        else:
            st.info("In-flight AI shepherd logs will appear here once an open position is evaluated.")
    except Exception as e:
        st.info(f"In-flight shepherd ledger rendering deferred: {type(e).__name__}")

    # ── 3.1c Multi-Asset Relative Strength & Fleet Hierarchy Matrix ───────────
    st.markdown("---")
    st.subheader("👑 Multi-Asset Relative Strength & Fleet Hierarchy Matrix (30-Minute Rotation)")
    try:
        raw_rs = query_system_state("ai_asset_rankings", "")
        if raw_rs:
            rs_data = json.loads(raw_rs)
            rs_leaders = rs_data.get("priority_leaders", [])
            rs_secondary = rs_data.get("secondary_tier", [])
            rs_avoid = rs_data.get("avoid_chop_list", [])
            rs_rat = rs_data.get("fleet_focus_rationale", "")
            rs_details = rs_data.get("asset_details", {})
            rs_time = utc_to_sast(rs_data.get("updated_at"))

            st.markdown(f"""
            <div style="background: #0d121f; border-left: 3px solid #00e5ff; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">
                <b style="color: #00e5ff;">👑 Strategic Fleet Focus:</b> <span style="color: #e2e8f0; font-size: 0.88rem;">{rs_rat}</span>
                <span style="color: #64748b; font-size: 0.75rem; float: right;">Evaluated: {rs_time}</span>
            </div>
            """, unsafe_allow_html=True)

            if rs_details:
                matrix_rows = []
                for sym_k, det_v in rs_details.items():
                    t_val = det_v.get("tier", "SECONDARY")
                    t_badge = "⭐ LEADER" if t_val == "LEADER" else ("🚫 AVOID / CHOP" if t_val == "AVOID" else "⏳ SECONDARY")
                    matrix_rows.append({
                        "Asset": sym_k,
                        "AI Fleet Tier": t_badge,
                        "Edge Score": f"{det_v.get('edge_score', 50)} / 100",
                        "Market Posture": det_v.get("posture", "NEUTRAL"),
                        "Tactical Microstructure Notes": det_v.get("notes", "")
                    })
                # Sort leaders first, then secondary, then avoid
                matrix_df = pd.DataFrame(matrix_rows)
                sort_order = {"⭐ LEADER": 0, "⏳ SECONDARY": 1, "🚫 AVOID / CHOP": 2}
                matrix_df["_sort"] = matrix_df["AI Fleet Tier"].map(sort_order).fillna(1)
                matrix_df = matrix_df.sort_values(by=["_sort", "Edge Score"], ascending=[True, False]).drop(columns=["_sort"])
                st.dataframe(matrix_df, width="stretch", hide_index=True)
        else:
            st.info("Relative strength matrix initializing...")
    except Exception as rs_ui_err:
        st.info(f"Relative strength matrix synchronizing: {rs_ui_err}")

    # ── 3.1d Sovereign Bitcoin Anchor & Contagion Defense Telemetry ──────────
    st.markdown("---")
    st.subheader("⚓ Sovereign Bitcoin Anchor & Cross-Market Contagion Telemetry")
    try:
        raw_btc_anchor = query_system_state("ai_btc_anchor", "")
        if raw_btc_anchor:
            btc_a_data = json.loads(raw_btc_anchor)
            b_state = btc_a_data.get("btc_anchor_state", "HEALTHY_EXPANSION")
            b_pause = bool(btc_a_data.get("alt_long_pause", False))
            b_risk = num(btc_a_data.get("contagion_risk_score"), 20)
            b_dir = btc_a_data.get("anchor_directive", "Normal operations.")
            b_pivot = btc_a_data.get("key_btc_pivot", 84500.0)
            b_time = utc_to_sast(btc_a_data.get("updated_at"))

            b_color = "#00ff66" if not b_pause else "#ff003c"
            b_gate = "🟢 PERMITTED (Open Runway)" if not b_pause else "⏸️ PAUSED (Contagion Defense Active)"

            st.markdown(f"""
            <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {b_color}; border-radius: 6px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <b style="color: #ffffff; font-size: 1.0rem;">Bitcoin Sovereign State: <span style="color: #00e5ff;">{b_state}</span></b>
                        <span style="color: #64748b; font-size: 0.75rem; margin-left: 12px;">Evaluated at {b_time}</span>
                    </div>
                    <div>
                        <span style="background: {'rgba(0,255,102,0.15)' if not b_pause else 'rgba(255,0,60,0.15)'}; color: {b_color}; border: 1px solid {b_color}; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">
                            ALT LONG GATE: {b_gate}
                        </span>
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.6; margin-bottom: 8px;">
                    Contagion Risk Meter: <b style="color: {'#00ff66' if b_risk < 40 else ('#ff9900' if b_risk < 70 else '#ff003c')};">{b_risk:.0f} / 100</b> | Key Pivot Reference: <b style="color: #e2e8f0;">${num(b_pivot, 0.0):,.0f}</b>
                </div>
                <div style="font-size: 0.8rem; color: #cbd5e1; background: rgba(255,255,255,0.02); padding: 10px 14px; border-radius: 4px; border-left: 2px solid rgba(255,255,255,0.1);">
                    <i>"{b_dir}"</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Bitcoin Anchor telemetry scheduled.")
    except Exception as btc_ui_err:
        st.info(f"Bitcoin anchor synchronizing: {btc_ui_err}")

    # ── Institutional Liquidity Radar & SFP Scanner ─────────────────────────────
    st.markdown("---")
    st.subheader("📡 Institutional Liquidity Radar & SFP Scanner (15m Multi-Asset Sweeps)")
    raw_radar = query_system_state("ai_liquidity_radar", "{}")
    radar_data = {}
    try:
        if raw_radar and raw_radar != "{}":
            radar_data = json.loads(raw_radar)
    except Exception:
        pass

    if radar_data:
        m_summary = radar_data.get("market_summary", "Multi-asset liquidity scanner active.")
        rad_time = utc_to_sast(radar_data.get("updated_at"))
        st.markdown(f"""
        <div style="background: #0d121f; border-left: 3px solid #00e5ff; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px;">
            <b style="color: #00e5ff;">🌊 Multi-Asset Structural Alignment:</b> <span style="color: #e2e8f0;">{m_summary}</span>
            <span style="color: #64748b; font-size: 0.75rem; float: right;">Updated: {rad_time}</span>
        </div>
        """, unsafe_allow_html=True)

        detected = radar_data.get("detected_patterns", [])
        if detected:
            rows_pat = []
            for p in detected:
                dir_badge = "🟢 LONG" if p.get("direction") == "LONG" else "🔴 SHORT"
                rows_pat.append({
                    "Asset": p.get("symbol", "N/A"),
                    "Pattern": p.get("pattern_type", "SFP"),
                    "Bias": dir_badge,
                    "TF": p.get("timeframe", "15m"),
                    "Quality": f"{p.get('quality_score', 80)}%",
                    "Key Level": f"${num(p.get('key_level'), 0.0):,.2f}" if p.get("key_level") else "---",
                    "Institutional Thesis": p.get("structural_thesis", "")
                })
            df_pat = pd.DataFrame(rows_pat)
            st.dataframe(df_pat, width="stretch", hide_index=True)
        else:
            st.info("No high-conviction liquidity sweep patterns active in current 15m window.")
    else:
        st.info("15-minute Institutional Liquidity Radar initializing...")

    # ── AI Chief Strategist Asset Hierarchy Leaderboard ───────────────────────
    st.markdown("---")
    st.subheader("👑 AI Chief Strategist: Fleet Asset Hierarchy & Leadership (2-Hour Alpha Ranking)")
    raw_ranking = query_system_state("ai_asset_rankings", "{}")
    rank_data = {}
    try:
        if raw_ranking and raw_ranking != "{}":
            rank_data = json.loads(raw_ranking)
    except Exception:
        pass

    if rank_data:
        r_time = utc_to_sast(rank_data.get("updated_at"))
        r_rationale = rank_data.get("fleet_focus_rationale", "Fleet operating under dynamic asset hierarchy.")
        leaders = rank_data.get("priority_leaders", [])
        secondary = rank_data.get("secondary_tier", [])
        avoid_list = rank_data.get("avoid_chop_list", [])

        leaders_html = " ".join([f"<span style='background: rgba(0,255,102,0.15); color: #00ff66; border: 1px solid #00ff66; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 6px;'>{a}</span>" for a in leaders]) or "None"
        secondary_html = " ".join([f"<span style='background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid #00e5ff; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 6px;'>{a}</span>" for a in secondary]) or "None"
        avoid_html = " ".join([f"<span style='background: rgba(255,0,60,0.15); color: #ff003c; border: 1px solid #ff003c; padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-right: 6px;'>{a}</span>" for a in avoid_list]) or "None"

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="color: #94a3b8; font-size: 0.85rem;">🏛️ <b>Chief Strategist Posture:</b> <span style="color: #e2e8f0;">{r_rationale}</span></span>
                <span style="color: #64748b; font-size: 0.75rem;">Ranked: {r_time}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 10px;">
                <div style="background: rgba(0,255,102,0.05); border: 1px solid rgba(0,255,102,0.2); padding: 10px 12px; border-radius: 6px;">
                    <b style="color: #00ff66; font-size: 0.85rem;">🥇 Priority Trend Leaders:</b><br><div style="margin-top: 6px;">{leaders_html}</div>
                </div>
                <div style="background: rgba(0,229,255,0.05); border: 1px solid rgba(0,229,255,0.2); padding: 10px 12px; border-radius: 6px;">
                    <b style="color: #00e5ff; font-size: 0.85rem;">🥈 Secondary Tier Focus:</b><br><div style="margin-top: 6px;">{secondary_html}</div>
                </div>
                <div style="background: rgba(255,0,60,0.05); border: 1px solid rgba(255,0,60,0.2); padding: 10px 12px; border-radius: 6px;">
                    <b style="color: #ff003c; font-size: 0.85rem;">🚫 Avoid / High Chop:</b><br><div style="margin-top: 6px;">{avoid_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        asset_details = rank_data.get("asset_details", {})
        if asset_details:
            with st.expander("🔍 Detailed Multi-Asset Edge Scores & Posture Breakdown"):
                detail_rows = []
                for a_sym, a_info in asset_details.items():
                    detail_rows.append({
                        "Asset": a_sym,
                        "Tier": a_info.get("tier", "SECONDARY"),
                        "Edge Score": f"{a_info.get('edge_score', 50)}/100",
                        "Posture": a_info.get("posture", "NEUTRAL"),
                        "Strategist Notes": a_info.get("notes", "")
                    })
                st.dataframe(pd.DataFrame(detail_rows), width="stretch", hide_index=True)
    else:
        st.info("Chief Strategist 2-hour asset hierarchy ranking scheduled...")

    # ── Multi-Timeframe (MTF) Confluence Gate (15m + 1h + 4h) ─────────────────
    st.markdown("---")
    st.subheader("🌐 Institutional Multi-Timeframe (MTF) Confluence Matrix (15m + 1h + 4h)")
    raw_mtf = query_system_state("ai_mtf_confluence", "{}")
    mtf_data = {}
    try:
        if raw_mtf and raw_mtf != "{}":
            mtf_data = json.loads(raw_mtf)
    except Exception:
        pass

    if mtf_data:
        m_bias = mtf_data.get("fleet_bias", "NEUTRAL_MIXED")
        m_sum = mtf_data.get("market_summary", "Multi-timeframe structure analyzed across fleet.")
        m_time = utc_to_sast(mtf_data.get("updated_at"))
        bias_color = "#00ff66" if "BULL" in m_bias else ("#ff003c" if "BEAR" in m_bias else "#00e5ff")

        st.markdown(f"""
        <div style="background: #0d121f; border-left: 3px solid {bias_color}; padding: 12px 16px; border-radius: 6px; margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <b style="color: {bias_color}; font-size: 0.95rem;">🌐 Fleet Multi-Timeframe Bias: {m_bias}</b>
                <span style="color: #64748b; font-size: 0.75rem;">Updated: {m_time}</span>
            </div>
            <span style="color: #cbd5e1; font-size: 0.85rem;">{m_sum}</span>
        </div>
        """, unsafe_allow_html=True)

        c_matrix = mtf_data.get("confluence_matrix", {})
        if c_matrix:
            mtf_rows = []
            for s_name, s_info in c_matrix.items():
                c_score = float(s_info.get("confluence_score") or 50.0)
                s_align = str(s_info.get("alignment", "CHOP_CONSOLIDATION") or "CHOP_CONSOLIDATION")
                v_long = bool(s_info.get("veto_long", False))
                v_short = bool(s_info.get("veto_short", False))
                
                if v_long and v_short:
                    status_badge = "🛑 DUAL VETO"
                elif v_long:
                    status_badge = "🚫 LONG VETOED"
                elif v_short:
                    status_badge = "🚫 SHORT VETOED"
                else:
                    status_badge = "🟢 CLEAR"

                barrier = float(s_info.get("key_htf_barrier", 0.0) or 0.0)
                barrier_str = f"${barrier:,.2f}" if barrier > 0 else "---"

                mtf_rows.append({
                    "Asset": s_name,
                    "Confluence Score": f"{c_score:.0f}%",
                    "MTF Alignment": s_align,
                    "Entry Gate": status_badge,
                    "Key 4h Barrier": barrier_str,
                    "Structural Thesis": s_info.get("structural_rationale", "")
                })
            st.dataframe(pd.DataFrame(mtf_rows), width="stretch", hide_index=True)
        else:
            st.info("Fleet MTF matrix calculating...")
    else:
        st.info("Multi-Timeframe Confluence Gate (15m + 1h + 4h) initializing on next cycle...")

    # ── AI Delta-Neutral Hedging Oracle (Macro Volatility Shield) ──────────────
    st.markdown("---")
    st.subheader("🛡️ AI Delta-Neutral Hedging Oracle (Macro Volatility Shield)")
    raw_hedge = query_system_state("ai_hedging_oracle", "{}")
    hedge_data = {}
    try:
        if raw_hedge and raw_hedge != "{}":
            hedge_data = json.loads(raw_hedge)
    except Exception:
        pass

    if hedge_data:
        h_shield = hedge_data.get("risk_shield_level", "GREEN_SHIELD")
        h_req = hedge_data.get("hedge_required", False)
        h_asset = hedge_data.get("hedge_asset", "ETH/USDT")
        h_side = hedge_data.get("hedge_side", "NONE")
        h_target = float(hedge_data.get("target_short_notional", 0.0) or 0.0)
        h_rat = hedge_data.get("rationale", "Portfolio net exposure within safety boundaries.")
        h_time = utc_to_sast(hedge_data.get("updated_at"))

        shield_color = "#00ff66" if h_shield == "GREEN_SHIELD" else ("#ff9900" if h_shield == "YELLOW_HEDGE" else "#ff003c")
        shield_label = "GREEN SHIELD (NORMAL RISK)" if h_shield == "GREEN_SHIELD" else ("YELLOW HEDGE (DEFENSIVE SHIELD)" if h_shield == "YELLOW_HEDGE" else "RED EMERGENCY DELEVERAGE")

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {shield_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="font-weight: bold; color: {shield_color}; font-size: 1.0rem;">🛡️ {shield_label}</span>
                    <span style="background: {'rgba(255,0,60,0.15)' if h_req else 'rgba(0,255,102,0.15)'}; color: {'#ff003c' if h_req else '#00ff66'}; border: 1px solid {'#ff003c' if h_req else '#00ff66'}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.78rem; margin-left: 10px;">
                        {'SYNTHETIC SHORT HEDGE RECOMMENDED' if h_req else 'UNHEDGED (FULL ALPHA EXPOSURE)'}
                    </span>
                </div>
                <span style="color: #64748b; font-size: 0.75rem;">{h_time}</span>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 12px; margin-top: 8px;">
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.75rem;">HEDGE ASSET</span><br>
                    <b style="color: #ffffff; font-size: 0.95rem;">{h_asset} ({h_side})</b>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.75rem;">TARGET NOTIONAL</span><br>
                    <b style="color: #00e5ff; font-size: 0.95rem;">${h_target:.2f} USD</b>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.75rem;">ORACLE ADVISORY</span><br>
                    <span style="color: #e2e8f0; font-size: 0.82rem;">{h_rat}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("AI Delta-Neutral Hedging Oracle standing by (calibrating macro risk exposure)...")

    # ── BTC Cash-and-Carry Basis Oracle & Delta-Neutral Engine ────────────────
    st.markdown("---")
    st.subheader("🪙 BTC Cash-and-Carry Basis Oracle & Delta-Neutral Engine")
    raw_basis = query_system_state("basis_trader_telemetry", "{}")
    basis_data = {}
    try:
        if raw_basis and raw_basis != "{}":
            basis_data = json.loads(raw_basis)
    except Exception:
        pass

    if basis_data:
        # Every figure below is published per field: a measured reading is shown as a reading and
        # a key the engine never sent is shown as an absence. The live telemetry carries 5 keys for
        # a DISABLED engine, so most of these are absences — and `net_delta = 0.0` is a measured
        # zero that stays a reading.
        b_r = _basis_readings(basis_data)
        b_state_st = b_r["state"]
        b_reason_st = b_r["reason"]
        b_margin_mode_st = b_r["margin_mode"]
        b_timestamp_st = b_r["timestamp"]
        b_funding_st = b_r["funding_rate_8h"]
        b_yield_st = b_r["annualized_yield_pct"]
        b_hurdle_st = b_r["hurdle_rate_pct"]
        b_unwind_st = b_r["unwind_hurdle_pct"]
        b_alloc_st = b_r["capital_allocated_usd"]
        b_harvest_st = b_r["cumulative_funding_harvested"]
        b_net_delta_st = b_r["net_delta"]
        b_holding_st = b_r["holding_days"]
        b_min_holding_st = b_r["min_holding_days"]
        b_fee_target_st = b_r["fee_hurdle_target_usd"]
        b_fee_amort_st = b_r["fee_amortized_pct"]
        b_liq_price_st = b_r["liquidation_price"]
        b_dist_liq_st = b_r["distance_to_liq_pct"]
        b_min_liq_st = b_r["min_liq_distance_pct"]
        b_spread_st = b_r["basis_spread_usd"]
        b_disloc_st = b_r["basis_dislocation_pct"]

        # ── Published figures, one text variable per card, so no `{x['value']}` can format None.
        b_alloc_text = f"${b_alloc_st['value']:,.0f} USDT" if b_alloc_st["measured"] else "not measured"
        b_alloc_note = "" if b_alloc_st["measured"] else _absence_note(b_alloc_st["reason"])
        b_funding_text = f"{b_funding_st['value'] * 100:+.4f}%" if b_funding_st["measured"] else "not measured"
        b_funding_note = "" if b_funding_st["measured"] else _absence_note(b_funding_st["reason"])
        b_yield_text = f"{b_yield_st['value']:+.2f}%" if b_yield_st["measured"] else "not measured"
        b_yield_note = "" if b_yield_st["measured"] else _absence_note(b_yield_st["reason"])
        b_hurdle_text = f"{b_hurdle_st['value']:.1f}%" if b_hurdle_st["measured"] else "not measured"
        b_hurdle_note = "" if b_hurdle_st["measured"] else _absence_note(b_hurdle_st["reason"])
        b_spread_text = f"${b_spread_st['value']:+.2f}" if b_spread_st["measured"] else "not measured"
        b_spread_note = "" if b_spread_st["measured"] else _absence_note(b_spread_st["reason"])
        b_disloc_text = f"({b_disloc_st['value']:+.2f}%)" if b_disloc_st["measured"] else "not measured"
        b_disloc_note = "" if b_disloc_st["measured"] else _absence_note(b_disloc_st["reason"])
        b_harvest_text = f"${b_harvest_st['value']:+.4f} USD" if b_harvest_st["measured"] else "not measured"
        b_harvest_note = "" if b_harvest_st["measured"] else _absence_note(b_harvest_st["reason"])
        b_net_delta_text = f"${b_net_delta_st['value']:+.2f}" if b_net_delta_st["measured"] else "not measured"
        b_net_delta_note = "" if b_net_delta_st["measured"] else _absence_note(b_net_delta_st["reason"])
        b_fee_amort_text = f"{b_fee_amort_st['value']:.1f}%" if b_fee_amort_st["measured"] else "not measured"
        b_fee_amort_note = "" if b_fee_amort_st["measured"] else _absence_note(b_fee_amort_st["reason"])
        b_fee_target_text = f"${b_fee_target_st['value']:.2f}" if b_fee_target_st["measured"] else "not measured"
        b_fee_target_note = "" if b_fee_target_st["measured"] else _absence_note(b_fee_target_st["reason"])
        b_liq_price_text = f"${b_liq_price_st['value']:,.2f}" if b_liq_price_st["measured"] else "not measured"
        b_liq_price_note = "" if b_liq_price_st["measured"] else _absence_note(b_liq_price_st["reason"])
        b_dist_liq_text = f"+{b_dist_liq_st['value']:.1f}% Buffer" if b_dist_liq_st["measured"] else "not measured"
        b_dist_liq_note = "" if b_dist_liq_st["measured"] else _absence_note(b_dist_liq_st["reason"])
        b_margin_mode_text = b_margin_mode_st["text"].upper() if b_margin_mode_st["measured"] else "not measured"
        b_margin_mode_note = "" if b_margin_mode_st["measured"] else _absence_note(b_margin_mode_st["reason"])
        b_holding_text = f"{b_holding_st['value']:.1f}d" if b_holding_st["measured"] else "not measured"
        b_holding_note = "" if b_holding_st["measured"] else _absence_note(b_holding_st["reason"])
        b_min_holding_text = f"{b_min_holding_st['value']:.0f}d" if b_min_holding_st["measured"] else "not measured"
        b_min_holding_note = "" if b_min_holding_st["measured"] else _absence_note(
            b_min_holding_st["reason"])

        # Net-delta posture: the word "Neutral" used to be hardcoded beside the figure, so the
        # panel asserted a hedged book even when no delta was published. It is now derived from the
        # published delta only, and says nothing when there is none. The tolerance it is judged
        # against is this panel's own pick, so the verdict names it instead of reading as a
        # property of the book.
        if b_net_delta_st["measured"]:
            b_delta_posture = (
                f"Neutral (±${_BASIS_NEUTRAL_TOL_USD:.2f} panel threshold)"
                if abs(b_net_delta_st["value"]) <= _BASIS_NEUTRAL_TOL_USD
                else f"Not neutral (±${_BASIS_NEUTRAL_TOL_USD:.2f} panel threshold)")
            delta_color = "#00e5ff"
        else:
            b_delta_posture = "posture not measured"
            delta_color = "#94a3b8"

        # ── State badge. The old else-branch claimed the engine was idle and scanning for yield and
        # spread for EVERY state, including a payload that published none, so a silent engine was
        # described as hunting a trade. An unpublished state is now published as an absence.
        if b_state_st["measured"]:
            b_state_key = b_state_st["text"].strip().upper()
            if b_state_key in _BASIS_STATE_BADGES:
                status_color = _BASIS_STATE_BADGES[b_state_key][0]
                status_badge = _BASIS_STATE_BADGES[b_state_key][1]
            else:
                status_color = "#94a3b8"
                status_badge = f"STATE PUBLISHED, UNRECOGNISED: {html.escape(b_state_st['text'])}"
        else:
            status_color = "#94a3b8"
            status_badge = "STATE NOT MEASURED"

        # The reason is the engine's own claim, so it is published when present and missed when not.
        if b_reason_st["measured"]:
            b_reason_line = (f"Engine-reported reason: "
                             f"<b style='color: #e2e8f0;'>{html.escape(b_reason_st['text'])}</b>")
        else:
            b_reason_line = ("Engine-reported reason: <b style='color: #94a3b8;'>not measured</b>"
                             + _absence_note(b_reason_st["reason"]))

        # ── Telemetry age: the payload's own timestamp is a measurement, so a 35-day-old row is
        # labelled stale instead of being allowed to look current.
        if b_timestamp_st["measured"]:
            b_age_days = b_r["age_days"]
            b_is_stale = bool(b_r["stale"])
            b_time_color = "#ffcc00" if b_is_stale else "#64748b"
            b_rendered_ts = utc_to_sast(b_timestamp_st["text"])
            if b_age_days is None:
                b_time_text = f"{b_rendered_ts} · age unknown"
            elif b_is_stale:
                b_time_text = f"{b_rendered_ts} · {b_age_days:.0f}d old · ⚠️ STALE TELEMETRY"
            else:
                b_time_text = f"{b_rendered_ts} · {b_age_days:.0f}d old"
        else:
            b_age_days = None
            b_is_stale = False
            b_time_color = "#94a3b8"
            b_time_text = "timestamp not measured"

        # ── Derived verdicts. Each one guards on the measured state of every reading it uses.
        # `yield_color` is a combined verdict — it compares the yield against two thresholds — so
        # it is legitimately bundled on all three states; a field's own text never is.
        if b_yield_st["measured"] and b_hurdle_st["measured"] and b_unwind_st["measured"]:
            yield_color = "#00ff66" if b_yield_st["value"] >= b_hurdle_st["value"] else (
                "#ff9900" if b_yield_st["value"] > b_unwind_st["value"] else "#ff003c")
        else:
            yield_color = "#94a3b8"
        if b_funding_st["measured"]:
            funding_color = "#00ff66" if b_funding_st["value"] >= 0.0 else "#ff003c"
        else:
            funding_color = "#94a3b8"
        # The liquidation guard's amber rung is the minimum the ENGINE publishes for itself
        # (`min_liq_distance_pct`, basis_trader.py:836 — the same figure it checks its own distance
        # against at `:315`), so the printed threshold is the operative one. The panel's own
        # `_BASIS_MIN_BUFFER_PCT` only stands in when nothing was published, and the label says
        # which of the two is in force. Green is always the panel's own safe-clear rung and never
        # sits below the threshold in force.
        b_buffer_threshold_pct = (b_min_liq_st["value"] if b_min_liq_st["measured"]
                                  else _BASIS_MIN_BUFFER_PCT)
        # B-7: that green rung used to live only in the code, so the all-clear was earned by a gate
        # the operator could not see while the header comment claimed it was labelled. It is printed
        # beside the threshold it is measured against, and it is named as the panel's own.
        b_buffer_label = (
            f"engine min-buffer threshold: {b_buffer_threshold_pct:.1f}%"
            if b_min_liq_st["measured"] else
            f"panel min-buffer threshold: {_BASIS_MIN_BUFFER_PCT:.0f}% (panel's own — the engine "
            f"published no threshold)") + (
            f" · panel safe-clear rung: {_BASIS_SAFE_BUFFER_PCT:.0f}% (panel's own)")
        if b_dist_liq_st["measured"]:
            b_buffer_guard_ok = (b_dist_liq_st["value"] >= _BASIS_SAFE_BUFFER_PCT
                                 and b_dist_liq_st["value"] >= b_buffer_threshold_pct)
            liq_color = "#00ff66" if b_buffer_guard_ok else (
                "#ff9900" if b_dist_liq_st["value"] >= b_buffer_threshold_pct else "#ff003c")
            margin_guard_color = "#00ff66" if b_buffer_guard_ok else "#ff9900"
        else:
            liq_color = "#94a3b8"
            margin_guard_color = "#94a3b8"
        if b_disloc_st["measured"]:
            disloc_color = "#00ff66" if abs(b_disloc_st["value"]) < 1.0 else "#ff9900"
        else:
            disloc_color = "#94a3b8"
        if b_fee_amort_st["measured"]:
            fee_color = "#00ff66" if b_fee_amort_st["value"] >= 100.0 else (
                "#00e5ff" if b_fee_amort_st["value"] >= 50.0 else "#94a3b8")
            fee_bar_width = min(100.0, max(
                0.0, (b_fee_amort_st["value"] / _BASIS_FEE_SCALE_PCT) * 100.0))
        else:
            fee_color = "#94a3b8"
            fee_bar_width = 0.0
        harvest_color = ("#00ff66" if b_harvest_st["value"] >= 0.0 else "#ff003c") \
            if b_harvest_st["measured"] else "#94a3b8"

        # One honest headline for a panel whose engine is parked: say how much of it was never
        # measured, instead of letting each blank look like a zero. The roster is the reader's own
        # key set minus the age fields, so a field added to `_basis_readings` is counted the day it
        # lands: a hand-kept tuple is the one way this headline could go on saying "15 of 18" about
        # a panel that had changed underneath it.
        b_economics_states = tuple(one_st for one_key, one_st in b_r.items()
                                   if one_key not in _BASIS_COVERAGE_EXCLUDED)
        b_unmeasured_count = sum(1 for one_st in b_economics_states if not one_st["measured"])
        b_field_count = len(b_economics_states)
        if b_unmeasured_count:
            b_coverage_line = (f"<span style='color: #94a3b8; font-size: 0.72rem;'>⚠️ "
                               f"{b_unmeasured_count} of {b_field_count} economics fields in this "
                               f"panel were never published by the basis engine — they are shown "
                               f"as not measured, not as readings.</span>")
        else:
            b_coverage_line = ""

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {status_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="font-weight: bold; color: {status_color}; font-size: 1.0rem;">🪙 BTC CASH-AND-CARRY BASIS ENGINE</span>
                    <span style="background: rgba(255,255,255,0.06); color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.78rem; margin-left: 10px;">
                        {status_badge}
                    </span>
                    <span style="background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; margin-left: 6px;">
                        Allocated: {b_alloc_text}{b_alloc_note}
                    </span>
                </div>
                <span style="color: {b_time_color}; font-size: 0.75rem;">{b_time_text}</span>
            </div>
            <div style="margin-bottom: 8px;">{b_reason_line}<br>{b_coverage_line}</div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 8px;">
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">8H FUNDING RATE</span><br>
                    <b style="color: {funding_color}; font-size: 0.95rem;">{b_funding_text}</b>{b_funding_note}
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">ANNUALIZED YIELD</span><br>
                    <b style="color: {yield_color}; font-size: 0.95rem;">{b_yield_text}</b>{b_yield_note} <span style="font-size:0.72rem; color:#94a3b8;">(Hurdle: {b_hurdle_text}{b_hurdle_note})</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">SPOT VS PERP SPREAD</span><br>
                    <b style="color: #ffffff; font-size: 0.95rem;">{b_spread_text}</b>{b_spread_note} <span style="font-size:0.72rem; color:{disloc_color};">{b_disloc_text}{b_disloc_note}</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">FUNDING HARVESTED</span><br>
                    <b style="color: {harvest_color}; font-size: 0.95rem;">{b_harvest_text}</b>{b_harvest_note}
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">NET MARKET DELTA</span><br>
                    <b style="color: {delta_color}; font-size: 0.95rem;">{b_net_delta_text}</b> · {b_delta_posture}{b_net_delta_note}
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06);">
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94a3b8; font-size: 0.72rem;">FEE HURDLE AMORTIZATION</span>
                        <span style="color: {fee_color}; font-size: 0.75rem; font-weight: bold;">{b_fee_amort_text}{b_fee_amort_note} · panel scale: {_BASIS_FEE_SCALE_PCT:.0f}%</span>
                    </div>
                    <div style="width: 100%; background: #1e293b; height: 6px; border-radius: 3px; margin: 5px 0; overflow: hidden;">
                        <div style="width: {fee_bar_width:.1f}%; background: {fee_color}; height: 100%; border-radius: 3px;"></div>
                    </div>
                    <span style="color: #cbd5e1; font-size: 0.72rem;">Harvested: {b_harvest_text} · Target: {b_fee_target_text}{b_fee_target_note} · Hold: {b_holding_text}{b_holding_note} / {b_min_holding_text}{b_min_holding_note}</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">PERP SHORT LIQUIDATION GUARD</span><br>
                    <b style="color: {liq_color}; font-size: 0.95rem;">{b_dist_liq_text}</b>{b_dist_liq_note}
                    <span style="color: #94a3b8; font-size: 0.72rem; margin-left: 6px;">(Liq: {b_liq_price_text}{b_liq_price_note})</span><br>
                    <span style="color: {margin_guard_color}; font-size: 0.72rem;">Margin mode: {b_margin_mode_text}{b_margin_mode_note} · {b_buffer_label}</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">EXECUTION STATE MACHINE</span><br>
                    <b style="color: #e2e8f0; font-size: 0.85rem;">Sequential Fill Protocol</b><br>
                    <span style="color: #94a3b8; font-size: 0.72rem;">Passive Maker Perp Short (15s gate) ➔ Atomic Spot Mirror (Zero Legging Delta)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("BTC Cash-and-Carry Basis Trader standing by (monitoring funding yield and spot-perp spread)...")

    st.markdown("---")
    st.subheader("🧠 Multi-Agent AI Telemetry & Neural Memory")
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.markdown("#### 📡 Recent AI Boardroom Decision Audits")
        df_debates = query_db(TRADING_LEDGER_DB, "SELECT timestamp, symbol, side, execute, conviction_score, red_team_objection, final_rationale FROM boardroom_debates ORDER BY id DESC LIMIT 8")
        if not df_debates.empty:
            df_debates['Time (SAST)'] = df_debates['timestamp'].apply(utc_to_sast)
            df_debates['Verdict'] = df_debates['execute'].apply(lambda x: "🟢 APPROVED" if x in (1, "1", True) else "🔴 VETOED")
            df_debates['Conviction'] = df_debates['conviction_score'].apply(lambda x: f"{num(x)*100:.0f}%")
            df_debates['Objection / Thesis'] = df_debates.apply(
                lambda r: str(r.get('red_team_objection') or r.get('final_rationale') or '')[:70] + "...",
                axis=1
            )
            show_deb = df_debates[['Time (SAST)', 'symbol', 'side', 'Verdict', 'Conviction', 'Objection / Thesis']].rename(columns={'symbol': 'Asset', 'side': 'Side'})
            st.dataframe(show_deb, width="stretch", hide_index=True)
        else:
            st.info("AI Boardroom decision ledger synchronizing...")

    with col_ai2:
        df_rule_total = query_db(TRADING_LEDGER_DB, "SELECT COUNT(*) as cnt FROM learned_rules")
        total_rules_active = int(df_rule_total.iloc[0]["cnt"]) if not df_rule_total.empty else 0
        st.markdown(f"#### 📚 Active Dynamic Rules ({total_rules_active} Rules in Memory)")
        df_rules = query_db(TRADING_LEDGER_DB, "SELECT category, rule_text, confidence, created_at FROM learned_rules ORDER BY rowid DESC LIMIT 6")
        if not df_rules.empty:
            for _, r in df_rules.iterrows():
                st.markdown(f"""
                <div style='background: #0d121f; border-left: 3px solid #00e5ff; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.85rem;'>
                    <b style='color: #00e5ff;'>[{r['category']}] ({num(r['confidence'])*100:.0f}% Conf)</b><br>
                    <span style='color: #e2e8f0;'>{r['rule_text']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"{total_rules_active} dynamic neural rules active in long-term memory.")

    st.markdown("---")
    st.subheader("🎯 AI Model Calibration & Reliability Scorecard (Prediction vs Realized Outcome)")
    try:
        from model_calibration_tracker import ModelCalibrationTracker
        calib_tracker = ModelCalibrationTracker()
        c_metrics = calib_tracker.compute_calibration_metrics()
        
        cal_col1, cal_col2, cal_col3 = st.columns(3)
        cal_values = _calibration_panel_values(c_metrics)
        with cal_col1:
            st.metric("Signals Evaluated", cal_values["signals_evaluated"])
        with cal_col2:
            st.metric("Historical Win Rate", cal_values["historical_win_rate"])
        with cal_col3:
            st.metric(
                "Brier Calibration Score",
                cal_values["brier_score"],
                delta=f"Grade: {cal_values['calibration_grade']}",
                # The grade is a label, not a direction: an arrow on it means nothing, and on an
                # absent grade it would dress "not measured" as a result.
                delta_color="off",
                delta_arrow="off",
            )

        if c_metrics.get("bins"):
            df_bins = pd.DataFrame(c_metrics["bins"])
            df_bins = df_bins.rename(columns={
                "bin_range": "Confidence Bin",
                "sample_count": "Samples",
                "avg_predicted_conf": "Avg Predicted Conf (%)",
                "observed_win_rate": "Observed Win Rate (%)",
                "calibration_gap": "Calibration Gap (%)"
            })
            st.dataframe(df_bins, width="stretch", hide_index=True)
    except Exception as c_err:
        st.info(f"Model calibration tracker syncing: {c_err}")

    st.markdown("---")
    df_lesson_total = query_db(TRADING_LEDGER_DB, "SELECT COUNT(*) as cnt FROM trade_lessons")
    total_lessons_db = int(df_lesson_total.iloc[0]["cnt"]) if not df_lesson_total.empty else 0
    st.subheader(f"📖 Post-Trade Forensic Lessons Extracted ({total_lessons_db} Lessons in Database)")
    df_lessons = query_db(TRADING_LEDGER_DB, "SELECT symbol, pnl, lesson, timestamp FROM trade_lessons ORDER BY id DESC LIMIT 5")
    if not df_lessons.empty:
        for _, l in df_lessons.iterrows():
            pnl_v = num(l['pnl'])
            color = "#00ff66" if pnl_v > 0 else "#ff003c"
            st.markdown(f"""
            <div style='background: #0d121f; border: 1px solid #1e293b; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;'>
                <span style='font-family: monospace; font-weight: bold; color: {color};'>🪙 {l['symbol']} ({'+$' if pnl_v>=0 else '-$'}{abs(pnl_v):.2f} USDT)</span> — 
                <span style='color: #cbd5e1; font-size: 0.9rem;'>{l['lesson']}</span>
                <span style='color: #64748b; font-size: 0.75rem; float: right;'>{utc_to_sast(l['timestamp'])}</span>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: RISK & MICROSTRUCTURE (Visual OFI Imbalance + Regime Deck + Veto Pulse)
# ═════════════════════════════════════════════════════════════════════════════
with tab_risk:
    st.subheader("🛡️ Institutional Risk & Quantitative Microstructure (V36.2)")
    
    rcol1, rcol2, rcol3 = st.columns(3)
    with rcol1:
        st.markdown("#### 🔒 Margin, Leverage & Allocation")
        lev_state = _published_number(query_system_state("recommended_leverage", ""), "no recommended leverage published")
        lev_text = f"{lev_state['value']:g}x" if lev_state["measured"] else "not measured"
        lev_note = "" if lev_state["measured"] else _absence_note(lev_state["reason"])
        st.write(f"• **Dynamic Leverage:** {lev_text}{lev_note} *(regime classifier)*", unsafe_allow_html=True)
        st.write(f"• **Core Allocation:** 2 Reserved Slots for BTC/ETH")
        st.write(f"• **Altcoin Ceiling:** 3 Coins Max Concurrent (out of {max_coin_slots})")
        st.write(f"• **Global Margin Limit:** {margin_limit_pct:.1f}% Ceiling (30% Cash Buffer)")
        st.write(f"• **Max Entries per Coin:** 2 Positions Max")

    with rcol2:
        st.markdown("#### 🌊 Macro & News Event Gating")
        df_macro = query_db(TRADING_LEDGER_DB, "SELECT * FROM macro_market_context ORDER BY id DESC LIMIT 1")
        macro_row = df_macro.iloc[0] if not df_macro.empty else None
        no_row = "no macro telemetry row recorded"
        spy_state = _published_number(macro_row.get("spy_price") if macro_row is not None else None, no_row)
        uup_state = _published_number(macro_row.get("uup_price") if macro_row is not None else None, no_row)
        dxy_state = _published_number(macro_row.get("dxy_price") if macro_row is not None else None, no_row)
        raw_regime = str(macro_row.get("macro_regime") or "").strip() if macro_row is not None else ""
        regime_state = _published_state(raw_regime, no_row)

        spy_text = f"${spy_state['value']:,.2f}" if spy_state["measured"] else "not measured"
        spy_note = "" if spy_state["measured"] else _absence_note(spy_state["reason"])
        dxy_text = f"{dxy_state['value']:.2f}" if dxy_state["measured"] else "not measured"
        dxy_note = "" if dxy_state["measured"] else _absence_note(dxy_state["reason"])
        uup_text = f"${uup_state['value']:,.2f}" if uup_state["measured"] else "not measured"
        uup_note = "" if uup_state["measured"] else _absence_note(uup_state["reason"])
        st.write(f"• **S&P 500 (SPY):** {spy_text}{spy_note}", unsafe_allow_html=True)
        st.write(f"• **USD Index (DXY):** {dxy_text}{dxy_note}", unsafe_allow_html=True)
        st.write(f"• **Invesco DB US Dollar (UUP):** {uup_text}{uup_note}", unsafe_allow_html=True)
        # Three cases, and they must not collapse into each other: a real regime is a reading
        # and is published as one; "UNKNOWN" is the sentinel recording that it could not
        # resolve the regime, which is not a measured regime and not a neutral one either; a
        # missing value is an absence that names its reason.
        if not regime_state["measured"]:
            st.write(f"• **Macro Regime:** <b style='color: #94a3b8;'>not measured</b>{_absence_note(regime_state['reason'])}", unsafe_allow_html=True)
        elif regime_state["text"].upper() == "UNKNOWN":
            st.write("• **Macro Regime:** <b style='color: #94a3b8;'>UNKNOWN (macro telemetry unresolved)</b>", unsafe_allow_html=True)
        else:
            # A measured regime is a reading, and the reading is database text: escape it like the
            # gate reason below rather than letting markdown metacharacters become formatting.
            st.write(f"• **Macro Regime:** {html.escape(regime_state['text'])}")

        news_state = _published_flag(query_system_state("news_event_active", ""), "no news gate state published")
        gate_reason = str(query_system_state("news_event_reason", "") or "").strip()
        news_text = ("ACTIVE" if news_state["flag"] else "CLEAR") if news_state["measured"] else "not measured"
        news_note = "" if news_state["measured"] else _absence_note(news_state["reason"])
        gate_detail = f" — {html.escape(gate_reason)}" if (news_state["measured"] and gate_reason) else ""
        st.write(
            f"• **News Calendar Gate:** {news_text}{gate_detail}{news_note} "
            f"(window −{int(config.PRE_NEWS_MINUTES)}m/+{int(config.POST_NEWS_MINUTES)}m around FOMC/CPI/NFP)",
            unsafe_allow_html=True,
        )
        st.write(f"• **Hurst Exponent Filter:** Hurst >= {config.HURST_TREND_FLOOR} for Trend Breakouts")

    with rcol3:
        st.markdown("#### 🔬 Microstructure Veto Pulse")
        last_sniper_veto = query_system_state("last_veto_sniper", "None")
        market_state = _published_state(query_system_state("market_regime_label", ""), "no market regime label published")
        warden_state = _published_state(query_system_state("warden_60m_regime", ""), "no 60m warden regime published")
        di_state = _optional_float(query_system_state("market_regime_di_spread", ""))
        macro4h_state = _published_state(query_system_state("macro_trend_4h", ""), "no 4h macro trend published")

        reg_color = _stance_color(market_state["text"], market_state["measured"],
                                  ("BULL_TREND", "BULLISH", "TRENDING"), ("BEAR_TREND", "BEARISH"))
        warden_color = _stance_color(warden_state["text"], warden_state["measured"],
                                     ("LONG_ONLY",), ("SHORT_ONLY",))
        # L-1 FIX: Escape all DB/LLM-derived text before interpolation into unsafe_allow_html=True
        safe_market_label = html.escape(market_state["text"])
        safe_warden_regime = html.escape(warden_state["text"])
        safe_macro_4h = html.escape(macro4h_state["text"])
        safe_last_veto = html.escape(str(last_sniper_veto))
        di_text = f"{di_state:+.2f}" if di_state is not None else "not measured"
        di_color = reg_color if di_state is not None else "#94a3b8"
        market_note = "" if market_state["measured"] else _absence_note(market_state["reason"])
        warden_note = "" if warden_state["measured"] else _absence_note(warden_state["reason"])
        di_note = "" if di_state is not None else _absence_note("no DI spread published")
        macro_note = "" if macro4h_state["measured"] else _absence_note(macro4h_state["reason"])
        st.write(f"• **Market Regime Label:** <b style='color: {reg_color};'>{safe_market_label}</b> <span style='color: #94a3b8; font-size: 0.8rem;'>(4H: {safe_macro_4h})</span>{macro_note}", unsafe_allow_html=True)
        st.write(f"• **DI Polarity Spread:** <b style='color: {di_color};'>{html.escape(di_text)}</b>{di_note}", unsafe_allow_html=True)
        st.write(f"• **60m Warden Bias:** <b style='color: {warden_color};'>{safe_warden_regime}</b>{warden_note}", unsafe_allow_html=True)
        st.write(f"• **Volatility Cascade Risk:** <b style='color: #94a3b8;'>not measured</b> <span style='color: #94a3b8; font-size: 0.8rem;'>(no cascade telemetry is published)</span>", unsafe_allow_html=True)
        veto_preview = safe_last_veto[:40] + ("..." if len(safe_last_veto) > 40 else "")
        st.write(f"• **Last Filter Gate:** <span style='color: #94a3b8; font-size: 0.8rem;'>{veto_preview}</span>", unsafe_allow_html=True)

    # P1: Visual OFI Depth Imbalance Meter & Regime Strength Deck
    st.markdown("---")
    st.markdown("#### 🔬 Real-Time Order Flow Imbalance (OFI) & Regime Strength Deck")

    df_micro = query_db(TRADING_LEDGER_DB, "SELECT symbol, liquidity_imbalance, bid_wall_price, ask_wall_price, timestamp FROM oracle_microstructure ORDER BY id DESC LIMIT 1")
    micro_row = df_micro.iloc[0] if not df_micro.empty else None
    no_ofi = "no oracle microstructure row recorded"
    imb_state = _published_number(micro_row.get("liquidity_imbalance") if micro_row is not None else None, no_ofi)
    bid_state = _published_number(micro_row.get("bid_wall_price") if micro_row is not None else None, no_ofi)
    ask_state = _published_number(micro_row.get("ask_wall_price") if micro_row is not None else None, no_ofi)
    raw_sym = str(micro_row.get("symbol") or "").strip() if micro_row is not None else ""
    micro_sym = raw_sym if raw_sym else "unknown symbol"
    safe_micro_sym = html.escape(micro_sym)

    # Visual bipolar OFI bar calculation (-1.0 to +1.0 mapped to 0% to 100%). The bar's 50%
    # position IS the "balanced" assertion, so a marker parked there is the same false
    # all-clear as the BALANCED LIQUIDITY label, drawn instead of written: when the reading
    # is absent the marker is omitted entirely and the verdict reads NOT MEASURED.
    if imb_state["measured"]:
        imb_v = imb_state["value"]
        marker_pct = max(0.0, min(100.0, ((imb_v + 1.0) / 2.0) * 100.0))
        imb_color = "#00ff66" if imb_v > 0.05 else ("#ff003c" if imb_v < -0.05 else "#94a3b8")
        bias_label = "AGGRESSIVE BUY PRESSURE" if imb_v > 0.10 else ("AGGRESSIVE SELL PRESSURE" if imb_v < -0.10 else "BALANCED LIQUIDITY")
        ofi_text = f"{imb_v:+.3f}"
        ofi_note = ""
        marker_html = (
            f"<div style=\"position: absolute; left: {marker_pct}%; top: -4px; width: 8px; height: 24px; "
            f"background: #ffffff; border: 1px solid #000; border-radius: 4px; "
            f"transform: translateX(-50%); box-shadow: 0 0 10px #ffffff;\"></div>"
        )
    else:
        imb_color = "#94a3b8"
        bias_label = "NOT MEASURED"
        ofi_text = "not measured"
        ofi_note = _absence_note(imb_state["reason"])
        marker_html = ""

    bid_text = f"${bid_state['value']:,.2f}" if bid_state["measured"] else "not measured"
    bid_color = "#00ff66" if bid_state["measured"] else "#94a3b8"
    bid_note = "" if bid_state["measured"] else _absence_note(bid_state["reason"])
    ask_text = f"${ask_state['value']:,.2f}" if ask_state["measured"] else "not measured"
    ask_color = "#ff003c" if ask_state["measured"] else "#94a3b8"
    ask_note = "" if ask_state["measured"] else _absence_note(ask_state["reason"])

    st.markdown(f"""
    <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
            <span style="color: #ff003c; font-weight: 700;">🔴 SELL PRESSURE (-1.0)</span>
            <span style="color: {imb_color}; font-weight: 800; font-size: 1.05rem;">
                {safe_micro_sym} OFI: {ofi_text} [{bias_label}]{ofi_note}
            </span>
            <span style="color: #00ff66; font-weight: 700;">🟢 BUY PRESSURE (+1.0)</span>
        </div>
        <div style="position: relative; width: 100%; height: 16px; background: linear-gradient(90deg, rgba(255,0,60,0.6) 0%, rgba(30,41,59,0.7) 50%, rgba(0,255,102,0.6) 100%); border-radius: 8px;">
            {marker_html}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
            <span>🟢 Major Bid Wall: <b style="color: {bid_color};">{bid_text}</b>{bid_note}</span>
            <span>⚡ Stream: <b style="color: #00e5ff;">100ms L2 Order Book</b></span>
            <span>🔴 Major Ask Wall: <b style="color: {ask_color};">{ask_text}</b>{ask_note}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 🔬 Quantitative Microstructure & Sizing Telemetry (Scouted Alpha Engine)
    # Every one of these eleven readings is measured or absent, per key. A missing key is an
    # absence and never a zero: `float(x or 0.0)` invented 0.0 for a key nobody wrote, and
    # `float(x or 0.5)` overwrote a measured 0.0 percentile with the stand-in it was standing
    # in for. A recorded zero — including a measured -0.0 — is a fact and keeps its sign.
    ms_json = None
    try:
        if os.path.exists(MICROSTRUCTURE_STATE_FILE):
            with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f_ms:
                ms_json = json.load(f_ms)
    except Exception:
        ms_json = None
    ms_ok = isinstance(ms_json, dict)
    no_ms = "microstructure telemetry is not readable"

    mlofi_state = _published_number(ms_json.get("mlofi") if ms_ok else None,
                                     no_ms if not ms_ok else "no mlofi in microstructure telemetry")
    micro_p_state = _published_number(ms_json.get("micro_price") if ms_ok else None,
                                       no_ms if not ms_ok else "no micro_price in microstructure telemetry")
    micro_bps_state = _published_number(ms_json.get("micro_price_bps") if ms_ok else None,
                                         no_ms if not ms_ok else "no micro_price_bps in microstructure telemetry")
    vpin_state = _published_number(ms_json.get("vpin") if ms_ok else None,
                                    no_ms if not ms_ok else "no vpin in microstructure telemetry")
    hawkes_spoof_state = _published_number(ms_json.get("hawkes_spoof_index") if ms_ok else None,
                                            no_ms if not ms_ok else "no hawkes_spoof_index in microstructure telemetry")
    hawkes_rho_state = _published_number(ms_json.get("hawkes_rho") if ms_ok else None,
                                          no_ms if not ms_ok else "no hawkes_rho in microstructure telemetry")
    hawkes_veto_state = _published_boolean(ms_json.get("hawkes_veto") if ms_ok else None,
                                            no_ms if not ms_ok else "no hawkes_veto in microstructure telemetry")
    kyle_pctile_state = _published_number(ms_json.get("kyle_lambda_pctile") if ms_ok else None,
                                           no_ms if not ms_ok else "no kyle_lambda_pctile in microstructure telemetry")
    lvr_tax_state = _published_number(ms_json.get("lvr_tax_bps") if ms_ok else None,
                                       no_ms if not ms_ok else "no lvr_tax_bps in microstructure telemetry")
    tox_halt_state = _published_boolean(ms_json.get("toxicity_halt") if ms_ok else None,
                                         no_ms if not ms_ok else "no toxicity_halt in microstructure telemetry")

    # Kelly Sizing fraction. A measured 0.0% is a measurement — the engine returns 0.0 when
    # there is no edge. Only an exception or a missing import is an absence, so this deck no
    # longer publishes a fabricated 15% when RiskManager cannot be built.
    kelly_state = {"value": None, "measured": False, "reason": "no kelly fraction published"}
    try:
        from risk_manager import RiskManager
        rm_dash = RiskManager()
        kelly_state = _published_number(rm_dash.calculate_empirical_kelly_fraction() * 100.0, "no kelly fraction published")
    except Exception:
        pass

    # Derived verdicts read the measured value only when there is one; an absent reading
    # gives the neutral grey and the NOT MEASURED badge, never a green all-clear.
    mlofi_text = f"{mlofi_state['value']:+.3f}" if mlofi_state["measured"] else "not measured"
    mlofi_color = ("#00ff66" if mlofi_state["value"] > 0.05 else ("#ff003c" if mlofi_state["value"] < -0.05 else "#94a3b8")) if mlofi_state["measured"] else "#94a3b8"
    vpin_text = f"{vpin_state['value']:.3f}" if vpin_state["measured"] else "not measured"
    vpin_color = ("#00ff66" if vpin_state["value"] < 0.45 else ("#f59e0b" if vpin_state["value"] < 0.70 else "#ff003c")) if vpin_state["measured"] else "#94a3b8"
    vpin_badge = ("🟢 BENIGN FLOW" if vpin_state["value"] < 0.45 else ("🟡 CAUTION FLOW" if vpin_state["value"] < 0.70 else "🔴 TOXIC CASCADE")) if vpin_state["measured"] else "NOT MEASURED"
    micro_color = ("#00ff66" if micro_bps_state["value"] > 0 else ("#ff003c" if micro_bps_state["value"] < 0 else "#94a3b8")) if micro_bps_state["measured"] else "#94a3b8"
    lvr_tax_text = f"{lvr_tax_state['value']:.1f} bps" if lvr_tax_state["measured"] else "not measured"

    if hawkes_veto_state["measured"] and hawkes_spoof_state["measured"]:
        hawkes_spoof_text = f"{hawkes_spoof_state['value']:.3f}"
        hawkes_color = "#ff003c" if (hawkes_veto_state["value"] or hawkes_spoof_state["value"] > 0.72) else ("#f59e0b" if hawkes_spoof_state["value"] > 0.45 else "#00ff66")
        hawkes_badge = "🔴 VETO TRIGGERED" if hawkes_veto_state["value"] else ("🟡 ELEVATED FLUX" if hawkes_spoof_state["value"] > 0.45 else "🟢 ORDER FLOW NORMAL")
    else:
        hawkes_spoof_text = "not measured"
        hawkes_color = "#94a3b8"
        hawkes_badge = "NOT MEASURED"
    hawkes_rho_text = f"{hawkes_rho_state['value']:.2f}" if hawkes_rho_state["measured"] else "not measured"

    if tox_halt_state["measured"] and kyle_pctile_state["measured"]:
        kyle_color = "#ff003c" if (tox_halt_state["value"] or kyle_pctile_state["value"] >= 0.90) else ("#f59e0b" if kyle_pctile_state["value"] >= 0.75 else "#00ff66")
        kyle_badge = "🔴 TOXICITY HALT" if tox_halt_state["value"] else f"Pctile: {kyle_pctile_state['value']*100:.0f}%"
    else:
        kyle_color = "#94a3b8"
        kyle_badge = "NOT MEASURED"
    kelly_text = f"{kelly_state['value']:.1f}%" if kelly_state["measured"] else "not measured"
    kelly_color = "#38bdf8" if kelly_state["measured"] else "#94a3b8"

    qcol1, qcol2, qcol3, qcol4, qcol5, qcol6 = st.columns(6)
    with qcol1:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {mlofi_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">🌊 Multi-Level OFI</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {mlofi_color}; font-family: 'JetBrains Mono', monospace;">{mlofi_text}</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">10-Level Decay (α=0.5)</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol2:
        # The old `> 0` test on the price was wrong in both directions: an absent price
        # rendered as a status message with no reason, and a measured 0 rendered as that same
        # absence. The reading decides the text, and only the reading.
        micro_str = f"${micro_p_state['value']:,.2f}" if micro_p_state["measured"] else "not measured"
        micro_note = "" if micro_p_state["measured"] else _absence_note(micro_p_state["reason"])
        micro_bias_text = f"{micro_bps_state['value']:+.1f} bps" if micro_bps_state["measured"] else "not measured"
        micro_bias_note = "" if micro_bps_state["measured"] else _absence_note(micro_bps_state["reason"])
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {micro_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">⚖️ Micro-Price Fair</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {micro_color}; font-family: 'JetBrains Mono', monospace;">{micro_str}{micro_note}</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">Bias: <b style="color: {micro_color};">{micro_bias_text}</b>{micro_bias_note}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {vpin_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">☣️ VPIN Toxicity</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {vpin_color}; font-family: 'JetBrains Mono', monospace;">{vpin_text}</div>
            <div style="color: {vpin_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">{vpin_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol4:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {hawkes_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">🦅 Hawkes Spoof Guard</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {hawkes_color}; font-family: 'JetBrains Mono', monospace;">{hawkes_spoof_text}</div>
            <div style="color: {hawkes_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">ρ={hawkes_rho_text} · {hawkes_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol5:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {kyle_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">λ Kyle / LVR Tax</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {kyle_color}; font-family: 'JetBrains Mono', monospace;">{lvr_tax_text}</div>
            <div style="color: {kyle_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">{kyle_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol6:
        kelly_note = "" if kelly_state["measured"] else _absence_note(kelly_state["reason"])
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {kelly_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">📐 Dynamic Kelly</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {kelly_color}; font-family: 'JetBrains Mono', monospace;">{kelly_text}{kelly_note}</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">Beta(13,7) Shrinkage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # ── Bayesian Markov Chain & Regime Transition Telemetry Deck ──────────────
    # Age first, because it is the finding: the row's own `updated_at` is the publish instant and
    # `config.REGIME_MAX_AGE_SEC` is the bound the fleet's own consumers enforce. Absent and stale
    # are DIFFERENT states -- the consumers refuse a stale row and have nothing at all to refuse
    # when no row was ever written -- and neither may wear a measured colour. No model is run to
    # fill a gap: the deck used to instantiate an untrained engine whose constructor is pure
    # (uniform prior, hardcoded transition matrix, no dataset), so what it printed was the model's
    # assumption about the world, not a measurement of this one.
    markov_raw, markov_published_at = query_system_state_row("markov_regime_telemetry")
    # The bound has to be PASSED, not merely documented in the comment above. Left at the signature
    # default, this call aged every row against the helper's own 900s fallback, so a fleet that
    # tightened `config.REGIME_MAX_AGE_SEC` kept refusing rows the deck still called live. One call
    # to `_regime_max_age_sec()` here keeps the deck and `inflight_shepherd.py:611` on one number.
    m_r = _markov_regime_readings(markov_raw, markov_published_at,
                                  max_age_sec=_regime_max_age_sec())
    m_dom_st = m_r["dominant_regime"]
    m_conf_st = m_r["confidence"]
    m_domdur_st = m_r["dominant_duration"]
    m_gate_st = m_r["is_tradable"]
    m_rsn_st = m_r["tradability_reason"]
    m_sizing_st = m_r["sizing_multiplier"]
    m_bd_st = m_r["breakdown_risk"]
    m_stat_st = m_r["equilibrium"]
    m_matrix_st = m_r["transition_matrix"]

    # The heading is the claim, so it is the first thing the age has to agree with. A row past the
    # bound the fleet refuses is not a live forward filter, and a row that measured nothing -- or
    # carries a stamp that cannot be read, or one that reads from the future -- is not one either.
    # The no-telemetry tail is three states, not one (finding #8): no row at all is the scheduler's
    # failure, a row that is not a telemetry object is the producer's, and a row that is an object
    # but published no fields is neither -- all in the deck's absence grey, none of them "Live".
    if m_r["heading_is_live"]:
        m_heading = "Live Forward Filter"
    elif m_r["stale"]:
        m_heading = "STALE Forward Filter — row past the fleet age bound"
    elif m_r["stamp_is_future"]:
        m_heading = "Forward Filter NOT MEASURED — publish stamp is ahead of this clock"
    elif not m_r["age_known"]:
        m_heading = "Forward Filter NOT MEASURED — no usable publish stamp"
    elif not m_r["row_written"]:
        m_heading = "Forward Filter NOT MEASURED — no row under this key"
    elif not m_r["row_is_telemetry_object"]:
        m_heading = ("Forward Filter NOT MEASURED — row present but not a regime telemetry "
                     "object")
    else:
        m_heading = "Forward Filter NOT MEASURED — row carries no regime telemetry"
    st.markdown(f"#### 🎲 Bayesian Markov Regime Engine & Transition Matrix ({m_heading})")

    # The age line, in the house shape: the stamp, the age, and the bound it blew through by name.
    # No age at all is an absence, so it wears the absence grey and says which failure left it blank;
    # a negative number is never printed as if it measured the row's youth (`age_sec` is None for
    # future stamps precisely so that this line cannot render one).
    if m_r["published_at"]["measured"]:
        m_stamp_text = utc_to_sast(m_r["published_at"]["text"])
        if m_r["age_sec"] is None:
            m_time_text = (f"published {m_stamp_text} · age not measured"
                           + _absence_note(m_r["age_absence_reason"]))
            m_time_color = "#94a3b8"
        elif m_r["stale"]:
            m_time_text = (f"{m_stamp_text} · {m_r['age_sec']:.0f}s old · ⚠️ STALE TELEMETRY — the "
                           f"fleet's consumers refuse this row past {m_r['max_age_sec']:.0f}s")
            m_time_color = "#ffcc00"
        else:
            m_time_text = (f"{m_stamp_text} · {m_r['age_sec']:.0f}s old · within the "
                           f"{m_r['max_age_sec']:.0f}s bound")
            m_time_color = "#64748b"
    else:
        m_time_text = ("row publish stamp: not measured"
                       + _absence_note(m_r["published_at"]["reason"]))
        m_time_color = "#94a3b8"

    # ── Published figures, one text variable per card, so no `{x['value']}` can format a None.
    # Each one keys on its OWN reading: an absent field says which key the engine never wrote, and
    # a recorded zero stays a reading (`BEAR_CASCADE: 0.0002` is published today).
    m_dom_text = m_dom_st["text"] if m_dom_st["measured"] else "not measured"
    m_dom_note = "" if m_dom_st["measured"] else _absence_note(m_dom_st["reason"])
    m_conf_text = f"{m_conf_st['value'] * 100:.1f}%" if m_conf_st["measured"] else "not measured"
    m_conf_note = "" if m_conf_st["measured"] else _absence_note(m_conf_st["reason"])
    m_domdur_text = f"{m_domdur_st['value']:.1f} Bars" if m_domdur_st["measured"] else "not measured"
    m_domdur_note = "" if m_domdur_st["measured"] else _absence_note(m_domdur_st["reason"])
    m_sizing_text = f"{m_sizing_st['value']:.2f}x" if m_sizing_st["measured"] else "not measured"
    m_sizing_note = "" if m_sizing_st["measured"] else _absence_note(m_sizing_st["reason"])
    m_bd_text = f"{m_bd_st['value'] * 100:.1f}%" if m_bd_st["measured"] else "not measured"
    m_bd_note = "" if m_bd_st["measured"] else _absence_note(m_bd_st["reason"])
    m_stat_text = ("/".join(f"{one * 100:.0f}%" for one in m_stat_st["rows"])
                   if m_stat_st["measured"] else "not measured")
    m_stat_note = "" if m_stat_st["measured"] else _absence_note(m_stat_st["reason"])
    m_matrix_note = "" if m_matrix_st["measured"] else _absence_note(m_matrix_st["reason"])

    # Which horizon readings the engine did not publish, named one by one. The bars below are drawn
    # per series, so the operator can see exactly which of the twelve probabilities is missing
    # rather than being shown a bar that stands at nothing.
    m_horizon_absent = tuple(
        m_r[one]["reason"] for one in m_r["field_names"]
        if one.startswith(("prob_", "dur_", "fc_", "stat_")) and not m_r[one]["measured"])
    m_horizon_note = ("<br><span style='color: #94a3b8; font-size: 0.70rem;'>horizon readings the "
                      "engine did not publish: " + html.escape("; ".join(m_horizon_absent))
                      + "</span>") if m_horizon_absent else ""

    # ── Derived colours. An absence wears the deck's grey and never a measured colour; a measured
    # reading keeps the colour it earned. `_MARKOV_*` rungs are the PANEL's own verdict thresholds
    # (the engine publishes a probability, not a label), so each one is printed beside its label.
    if m_dom_st["measured"]:
        m_dom_color = _stance_color(m_dom_st["text"], True, ("BULL_EXPANSION",), ("BEAR_CASCADE",))
    else:
        m_dom_color = "#94a3b8"
    if m_conf_st["measured"]:
        m_conf_color = "#f59e0b"
    else:
        m_conf_color = "#94a3b8"
    if m_domdur_st["measured"]:
        m_domdur_color = "#00e5ff"
    else:
        m_domdur_color = "#94a3b8"
    if m_sizing_st["measured"]:
        m_sizing_color = "#38bdf8"
    else:
        m_sizing_color = "#94a3b8"
    if m_stat_st["measured"]:
        m_stat_color = "#a855f7"
    else:
        m_stat_color = "#94a3b8"
    if m_matrix_st["measured"]:
        m_matrix_color = "#38bdf8"
    else:
        m_matrix_color = "#94a3b8"
    if m_bd_st["measured"]:
        m_bd_color = ("#ff003c" if m_bd_st["value"] >= _MARKOV_BD_ALARM_PCT
                      else ("#f59e0b" if m_bd_st["value"] >= _MARKOV_BD_WARN_PCT else "#00ff66"))
        m_bd_label = ("HIGH" if m_bd_st["value"] >= _MARKOV_BD_HIGH_PCT
                      else ("ELEV" if m_bd_st["value"] >= _MARKOV_BD_ELEV_PCT else "STABLE"))
    else:
        m_bd_color = "#94a3b8"
        m_bd_label = "NOT MEASURED"
    # The tradability gate is the deck's COMBINED verdict — "may this row be traded right now" — so
    # it is the one card allowed to bundle (brief:252), and what it bundles is its own field AND the
    # row's freshness, exactly as the fleet's four automated consumers do: `inflight_shepherd.py:614`
    # and `super_bot.py:733` both refuse on `age > max_age`. Two failures therefore reach it: a gate
    # nobody measured (the HUD used to go green while `super_bot.py:733` logs a key-missing warning
    # and skips), and a measured `true` carried by a row the fleet refuses. The second one is why the
    # badge cannot key on `measured` alone -- B-1 asks the heading AND every verdict derived from the
    # row to say the row is refused. A measured false keeps its alarm red whatever the stamp says:
    # that verdict is the engine's own, and dimming it would be the same sin in the other direction.
    if m_gate_st["measured"]:
        if not m_gate_st["value"]:
            m_tradable_color = "#ff003c"
            m_tradable_badge = "UNTRADEABLE CHURN"
        elif m_r["stale"]:
            m_tradable_color = "#ffcc00"
            m_tradable_badge = "TRADABLE (STALE ROW)"
        elif not m_r["age_known"]:
            m_tradable_color = "#ffcc00"
            m_tradable_badge = "TRADABLE (FRESHNESS NOT MEASURED)"
        else:
            m_tradable_color = "#00ff66"
            m_tradable_badge = "TRADABLE"
    else:
        m_tradable_color = "#94a3b8"
        m_tradable_badge = "GATE NOT MEASURED"
    m_gate_note = "" if m_gate_st["measured"] else _absence_note(m_gate_st["reason"])

    # The reason is the engine's own words, published when it said any and missed when it did not.
    if m_rsn_st["measured"]:
        m_reason_line = ("Engine-reported reason: <b style='color: #e2e8f0;'>"
                         f"{html.escape(m_rsn_st['text'])}</b>")
    else:
        m_reason_line = ("Engine-reported reason: not measured"
                         + _absence_note(m_rsn_st["reason"]))

    # One honest headline, counted from the reader's own roster rather than a tuple kept by hand.
    m_unmeasured_count = m_r["coverage"]["unmeasured"]
    m_field_count = m_r["coverage"]["total"]
    if m_unmeasured_count:
        m_coverage_line = (f"<span style='color: #94a3b8; font-size: 0.72rem;'>⚠️ "
                           f"{m_unmeasured_count} of {m_field_count} published fields in this deck "
                           f"were never measured by the regime engine — they are shown as not "
                           f"measured, not as readings.</span>")
    else:
        m_coverage_line = ""

    st.markdown(f"""
    <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_time_color}; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #cbd5e1; font-size: 0.78rem;">{m_reason_line}</span>
            <span style="color: {m_time_color}; font-size: 0.75rem; white-space: nowrap;">{m_time_text}</span>
        </div>
        <div style="margin-top: 6px;">{m_coverage_line}</div>
    </div>
    """, unsafe_allow_html=True)

    # Top 6 Markov & Playbook KPI Cards
    mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
    with mk1:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_dom_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Active State (π_t)</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {m_dom_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{m_dom_text}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Conf: <b style="color: {m_conf_color};">{m_conf_text}</b>{m_conf_note}</div>
        </div>
        """, unsafe_allow_html=True)

    with mk2:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_domdur_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Expected Duration</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {m_domdur_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{m_domdur_text}{m_domdur_note}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">E[D] = 1 / (1 - P_ii)</div>
        </div>
        """, unsafe_allow_html=True)

    with mk3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_tradable_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Tradability Gate</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: {m_tradable_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">[{m_tradable_badge}]{m_gate_note}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">E[D] ≥ {_MARKOV_FEE_SHIELD_BARS}b fee shield (panel label — the engine's own reason is above)</div>
        </div>
        """, unsafe_allow_html=True)

    with mk4:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_sizing_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Risk Multiplier</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {m_sizing_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{m_sizing_text}{m_sizing_note}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Continuous π-Scaling, bounded [0.20, 1.0] by the engine</div>
        </div>
        """, unsafe_allow_html=True)

    with mk5:
        m_bd_rungs = (f" · panel rungs: ≥{_MARKOV_BD_HIGH_PCT:.2f} HIGH, "
                      f"≥{_MARKOV_BD_ELEV_PCT:.2f} ELEV, below STABLE"
                      if m_bd_st["measured"] else "")
        m_bd_alarm = (f" · panel alarm at {_MARKOV_BD_ALARM_PCT:.2f}, warn at "
                      f"{_MARKOV_BD_WARN_PCT:.2f}" if m_bd_st["measured"] else "")
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_bd_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">4-Step Cascade Risk</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {m_bd_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{m_bd_text}{m_bd_note}</div>
            <div style="color: {m_bd_color}; font-size: 0.70rem; margin-top: 2px; font-weight: bold;">[{m_bd_label}] P(Cascade){m_bd_rungs}{m_bd_alarm}</div>
        </div>
        """, unsafe_allow_html=True)

    with mk6:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {m_stat_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Equilibrium (π*)</div>
            <div style="font-size: 0.95rem; font-weight: 800; color: {m_stat_color}; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{m_stat_text}{m_stat_note}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Chop/Bull/Bear</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 2-Column Plotly Visualizations: Transition Matrix Heatmap & Multi-Horizon Probabilities
    col_hm, col_bars = st.columns([1, 1])

    # A horizon series is drawn only when every probability in it was published. A bar at 0.0 would
    # publish an absence as a measured zero, so the series is left out and the skip is named.
    m_series_specs = (("Current π_t", "prob"), ("Forecast π_{t+4}", "fc"),
                      ("Equilibrium π*", "stat"))
    m_series_states = tuple(
        {"label": one_label,
         "values": [one[one_key]["value"] for one in m_r["by_state"]],
         "measured": all(one[one_key]["measured"] for one in m_r["by_state"])}
        for one_label, one_key in m_series_specs)
    m_drawn_series = tuple(one for one in m_series_states if one["measured"])
    m_series_colors = {"Current π_t": "#00e5ff", "Forecast π_{t+4}": "#a855f7",
                       "Equilibrium π*": "#64748b"}

    if m_matrix_st["measured"]:
        # The row says a matrix was published; it cannot say where the matrix came from.
        # `process_bars` emits `self.P.tolist()` -- the prior at `markov_regime.py:39-43` -- on the
        # short-frame path (`:408`) without calling `estimate_transition_matrix`, which runs only on
        # the >=15-bar path (`:416`), so no consumer can tell an estimated matrix from the default.
        m_matrix_html = (f"<div style=\"color: {m_matrix_color}; font-size: 0.72rem;\">Transition "
                         f"matrix the engine published (its estimating path is not recoverable "
                         f"from the row)</div>")
    else:
        m_matrix_html = ("<div style=\"color: #94a3b8; font-size: 0.72rem;\">Transition matrix not "
                         f"measured{m_matrix_note}</div>")
    if any(one["measured"] for one in m_series_states):
        m_bars_html = (f"<div style=\"color: #94a3b8; font-size: 0.72rem;\">Drawing "
                       f"{len(m_drawn_series)} of {len(m_series_states)} horizon series</div>")
    else:
        m_bars_html = ("<div style=\"color: #94a3b8; font-size: 0.72rem;\">Horizon bars not "
                       "measured</div>" + _absence_note(
                           "no complete horizon series was published to plot"))

    with col_hm:
        st.markdown(m_matrix_html, unsafe_allow_html=True)
        # The heatmap is the published matrix or nothing: there is no stand-in P left in this file,
        # and a prior plotted under an "estimated" title is the defect this deck was rebuilt to end.
        if m_matrix_st["measured"]:
            mat_P = np.array(m_matrix_st["rows"], dtype=np.float64)
            text_matrix = []
            for i, one_from in enumerate(m_r["by_state"]):
                row_text = []
                for j, one_to in enumerate(m_r["by_state"]):
                    val = mat_P[i, j]
                    if i == j:
                        e_d_text = (f"<span style='font-size:9px; color:#38bdf8;'>E[D]="
                                    f"{one_to['dur']['value']:.1f}b</span>"
                                    if one_to["dur"]["measured"] else
                                    "<span style='font-size:9px; color:#94a3b8;'>E[D] not "
                                    "measured</span>")
                        row_text.append(f"{val:.2f}<br>{e_d_text}")
                    else:
                        row_text.append(f"{val:.2f}")
                text_matrix.append(row_text)

            fig_hm = go.Figure(data=go.Heatmap(
                z=mat_P.tolist(),
                x=["To CHOP", "To BULL", "To BEAR"],
                y=["From CHOP", "From BULL", "From BEAR"],
                text=text_matrix,
                texttemplate="%{text}",
                textfont={"size": 12, "family": "JetBrains Mono"},
                colorscale=[
                    [0.0, "#0d121f"],
                    [0.25, "#1e293b"],
                    [0.5, "#0284c7"],
                    [0.75, "#00e5ff"],
                    [1.0, "#00ff66"]
                ],
                zmin=0.0,
                zmax=1.0,
                showscale=False
            ))
            fig_hm.update_layout(
                title={"text": "<b>3×3 Bayesian Transition Matrix P(S_j | S_i)</b>", "font": {"size": 13, "color": "#cbd5e1"}},
                template="plotly_dark",
                paper_bgcolor="#0d121f",
                plot_bgcolor="#0d121f",
                height=280,
                margin=dict(l=40, r=20, t=40, b=30),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_hm, width="stretch", config={"displayModeBar": False})

    with col_bars:
        st.markdown(m_bars_html + m_horizon_note, unsafe_allow_html=True)
        if m_drawn_series:
            fig_bars = go.Figure()
            for one_series in m_drawn_series:
                fig_bars.add_trace(go.Bar(
                    name=one_series["label"],
                    x=list(m_r["states"]),
                    y=one_series["values"],
                    marker_color=m_series_colors[one_series["label"]]
                ))

            # Add the entry hysteresis threshold line the panel draws every horizon against.
            fig_bars.add_hline(
                y=_MARKOV_HYSTERESIS_GATE,
                line_dash="dot",
                line_color="#ff9900",
                annotation_text=f"{_MARKOV_HYSTERESIS_GATE:.2f} Hysteresis Gate (panel's own)",
                annotation_position="top right",
                annotation_font={"size": 10, "color": "#ff9900"}
            )

            fig_bars.update_layout(
                title={"text": "<b>Markov Horizon: Posterior vs Forecast vs Equilibrium</b>", "font": {"size": 13, "color": "#cbd5e1"}},
                template="plotly_dark",
                paper_bgcolor="#0d121f",
                plot_bgcolor="#0d121f",
                barmode="group",
                height=280,
                margin=dict(l=30, r=20, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                yaxis=dict(range=[0, 1.0], gridcolor="#1e293b")
            )
            st.plotly_chart(fig_bars, width="stretch", config={"displayModeBar": False})

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)


    # ── Multi-Asset Fleet Microstructure & Toxicity Matrix ─────────────────────
    if os.path.exists(MICROSTRUCTURE_STATE_FILE):
        try:
            with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f_ms_all:
                ms_all_json = json.load(f_ms_all)
            symbols_map = ms_all_json.get("symbols", {})
            if symbols_map:
                st.markdown("##### 🌊 Fleet Multi-Level Microstructure & Toxicity Matrix")
                fleet_rows = []
                for sym_code, sym_info in sorted(symbols_map.items()):
                    s_disp = sym_code.replace("USDT", "/USDT")
                    # One row per symbol, one measurement per cell. A symbol entry that lacks a
                    # key publishes "not measured" in that cell and NOT MEASURED for its
                    # verdict; it does not publish 0.00 and a green label built on it.
                    s_absent = f"no {sym_code} microstructure telemetry published"
                    s_mlofi_st = _published_number(sym_info.get("mlofi"), s_absent)
                    s_micro_price_st = _published_number(sym_info.get("micro_price"), s_absent)
                    s_micro_bps_st = _published_number(sym_info.get("micro_price_bps"), s_absent)
                    s_vpin_st = _published_number(sym_info.get("vpin"), s_absent)
                    s_best_bid_st = _published_number(sym_info.get("best_bid"), s_absent)
                    s_best_ask_st = _published_number(sym_info.get("best_ask"), s_absent)
                    s_hawkes_spoof_st = _published_number(sym_info.get("hawkes_spoof_index"), s_absent)
                    s_hawkes_rho_st = _published_number(sym_info.get("hawkes_rho"), s_absent)
                    s_hawkes_veto_st = _published_boolean(sym_info.get("hawkes_veto"), s_absent)
                    s_kyle_pctile_st = _published_number(sym_info.get("kyle_lambda_pctile"), s_absent)
                    s_lvr_tax_st = _published_number(sym_info.get("lvr_tax_bps"), s_absent)
                    s_tox_halt_st = _published_boolean(sym_info.get("toxicity_halt"), s_absent)

                    if s_best_bid_st["measured"]:
                        s_bid_price = s_best_bid_st["value"]
                        s_bid_text = f"${s_bid_price:,.4f}" if s_bid_price < 10 else f"${s_bid_price:,.2f}"
                    else:
                        s_bid_text = "not measured"
                    if s_best_ask_st["measured"]:
                        s_ask_price = s_best_ask_st["value"]
                        s_ask_text = f"${s_ask_price:,.4f}" if s_ask_price < 10 else f"${s_ask_price:,.2f}"
                    else:
                        s_ask_text = "not measured"
                    if s_micro_price_st["measured"]:
                        s_micro_price_text = f"${s_micro_price_st['value']:,.4f}" if s_micro_price_st["value"] < 10 else f"${s_micro_price_st['value']:,.2f}"
                    else:
                        s_micro_price_text = "not measured"

                    flow_posture = ("🟢 Buyer Depth" if s_mlofi_st["value"] > 5.0 else ("🔴 Seller Depth" if s_mlofi_st["value"] < -5.0 else "⚪ Balanced")) if s_mlofi_st["measured"] else "NOT MEASURED"
                    if (s_hawkes_veto_st["measured"] and s_hawkes_spoof_st["measured"]
                            and s_hawkes_rho_st["measured"]):
                        h_stat = "🔴 VETO ACTIVE" if (s_hawkes_veto_st["value"] or s_hawkes_spoof_st["value"] > 0.72 or s_hawkes_rho_st["value"] >= 1.0) else ("🟡 Elevated" if s_hawkes_spoof_st["value"] > 0.45 else "🟢 Stationary")
                    else:
                        h_stat = "NOT MEASURED"
                    if s_tox_halt_st["measured"] and s_kyle_pctile_st["measured"]:
                        tox_stat = "🔴 HALT ACTIVE" if s_tox_halt_st["value"] else (f"🟢 Normal ({s_kyle_pctile_st['value']*100:.0f}%)" if s_kyle_pctile_st["value"] < 0.80 else f"🟡 High ({s_kyle_pctile_st['value']*100:.0f}%)")
                    else:
                        tox_stat = "NOT MEASURED"

                    fleet_rows.append({
                        "Asset": s_disp,
                        "Best Bid": s_bid_text,
                        "Best Ask": s_ask_text,
                        "Micro-Price (Fair Value)": s_micro_price_text,
                        "Spread Bias": f"{s_micro_bps_st['value']:+.2f} bps" if s_micro_bps_st["measured"] else "not measured",
                        "MLOFI (10-Level)": f"{s_mlofi_st['value']:+.2f}" if s_mlofi_st["measured"] else "not measured",
                        "VPIN (Toxicity)": f"{s_vpin_st['value']:.3f}" if s_vpin_st["measured"] else "not measured",
                        "Kyle Flow Risk": tox_stat,
                        "LVR Tax": f"{s_lvr_tax_st['value']:.1f} bps" if s_lvr_tax_st["measured"] else "not measured",
                        "Hawkes Spoof": f"{s_hawkes_spoof_st['value']:.3f}" if s_hawkes_spoof_st["measured"] else "not measured",
                        "Spectral Radius (ρ)": f"{s_hawkes_rho_st['value']:.2f}" if s_hawkes_rho_st["measured"] else "not measured",
                        "LOB Anti-Spoof": h_stat,
                        "Order Flow Bias": flow_posture
                    })
                
                df_fleet_micro = pd.DataFrame(fleet_rows)
                st.dataframe(df_fleet_micro, width="stretch", hide_index=True)
        except Exception:
            pass

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # 🔬 AI Order Flow Tape Whisperer Intel (Groq Sub-Second L2 Inference)
    raw_tape = query_system_state("order_flow_tape_intel", "{}")
    tape_data = {}
    try:
        if raw_tape and raw_tape != "{}":
            tape_data = json.loads(raw_tape)
    except Exception:
        pass

    if tape_data:
        t_cond = tape_data.get("tape_condition", "NORMAL_FLOW")
        t_side = tape_data.get("absorption_side", "NONE")
        t_veto = tape_data.get("execution_veto", "CLEAR")
        t_conf = num(tape_data.get("confidence"), 0.80) * 100.0
        t_head = tape_data.get("headline", "Order flow tape analyzed.")
        t_time = utc_to_sast(tape_data.get("updated_at"))

        veto_bg = "rgba(0,255,102,0.15)" if t_veto == "CLEAR" else "rgba(255,0,60,0.15)"
        veto_color = "#00ff66" if t_veto == "CLEAR" else "#ff003c"

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div>
                    <span style="font-weight: bold; color: #00e5ff; font-size: 0.95rem;">⚡ AI Tape Whisperer (Groq L2 Tape Reading):</span>
                    <span style="background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid #00e5ff; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-left: 8px;">{t_cond}</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 8px;">Absorption: <b style="color: #ffffff;">{t_side}</b></span>
                </div>
                <div>
                    <span style="background: {veto_bg}; color: {veto_color}; border: 1px solid {veto_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">
                        VETO: {t_veto} ({t_conf:.0f}% Conf)
                    </span>
                    <span style="color: #64748b; font-size: 0.75rem; margin-left: 10px;">{t_time}</span>
                </div>
            </div>
            <p style="color: #cbd5e1; font-size: 0.85rem; margin: 4px 0 0 0;">{t_head}</p>
        </div>
        """, unsafe_allow_html=True)

    # 👻 AI Order Book Ghost Wall & Spoofing Detector (Groq Sub-Second L2 Inference)
    raw_ghost = query_system_state("ai_ghost_walls", "{}")
    ghost_data = {}
    try:
        if raw_ghost and raw_ghost != "{}":
            ghost_data = json.loads(raw_ghost)
    except Exception:
        pass

    if ghost_data:
        g_spoof = ghost_data.get("spoof_detected", False)
        g_side = ghost_data.get("spoof_side", "NONE")
        g_adv = ghost_data.get("execution_advisory", "ALLOW")
        g_risk = float(ghost_data.get("manipulation_risk_pct") or 10.0)
        g_head = ghost_data.get("headline", "Order book queue analysis completed.") or "Order book queue analysis completed."
        g_time = utc_to_sast(ghost_data.get("updated_at"))

        g_border = "#ff003c" if g_spoof else "#00ff66"
        g_badge_bg = "rgba(255,0,60,0.15)" if g_spoof else "rgba(0,255,102,0.15)"
        g_badge_color = "#ff003c" if g_spoof else "#00ff66"
        g_title = f"⚠️ PREDATORY SPOOFING DETECTED ({g_side} SIDE)" if g_spoof else "🟢 CLEAN ORDER BOOK DYNAMICS (NO SPOOFING)"

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {g_border}; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div>
                    <span style="font-weight: bold; color: {g_border}; font-size: 0.95rem;">👻 AI Ghost Wall & Spoofing Radar (Groq L2 Ingestion):</span>
                    <span style="background: {g_badge_bg}; color: {g_badge_color}; border: 1px solid {g_badge_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-left: 8px;">{g_title}</span>
                </div>
                <div>
                    <span style="background: {g_badge_bg}; color: {g_badge_color}; border: 1px solid {g_badge_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">
                        ADVISORY: {g_adv} ({g_risk:.0f}% Risk)
                    </span>
                    <span style="color: #64748b; font-size: 0.75rem; margin-left: 10px;">{g_time}</span>
                </div>
            </div>
            <p style="color: #cbd5e1; font-size: 0.85rem; margin: 4px 0 8px 0;">{g_head}</p>
        </div>
        """, unsafe_allow_html=True)

        phantom_walls = ghost_data.get("phantom_walls", [])
        genuine_anchors = ghost_data.get("genuine_anchors", [])

        if phantom_walls or genuine_anchors:
            gw_col1, gw_col2 = st.columns(2)
            with gw_col1:
                st.markdown("<b style='color: #ff003c; font-size: 0.85rem;'>👻 Detected Phantom Spoof Walls (<250ms Residence)</b>", unsafe_allow_html=True)
                if phantom_walls:
                    pw_rows = [{"Side": str(w.get("side", "N/A") or "N/A"), "Price": f"${float(w.get('price') or 0.0):,.2f}", "Size": f"${float(w.get('size_usd') or 0.0):,.0f}", "Description": str(w.get("description", "") or "")} for w in phantom_walls]
                    st.dataframe(pd.DataFrame(pw_rows), width="stretch", hide_index=True)
                else:
                    st.caption("No phantom spoof walls detected in current depth scan.")
            with gw_col2:
                st.markdown("<b style='color: #00ff66; font-size: 0.85rem;'>🧱 Genuine Institutional Iceberg Anchors (>2500ms Residence)</b>", unsafe_allow_html=True)
                if genuine_anchors:
                    ga_rows = [{"Side": str(a.get("side", "N/A") or "N/A"), "Price": f"${float(a.get('price') or 0.0):,.2f}", "Size": f"${float(a.get('size_usd') or 0.0):,.0f}", "Description": str(a.get("description", "") or "")} for a in genuine_anchors]
                    st.dataframe(pd.DataFrame(ga_rows), width="stretch", hide_index=True)
                else:
                    st.caption("No persistent iceberg anchors detected.")

    # Quantitative Regime Deck Gauges (ADX + Hurst Exponent)
    df_regime = query_db(TRADING_DATA_DB, "SELECT * FROM market_regime ORDER BY id DESC LIMIT 1")
    df_oracle = query_db(TRADING_LEDGER_DB, "SELECT * FROM oracle_state ORDER BY id DESC LIMIT 1")

    readings = _regime_readings(
        df_regime.iloc[0] if not df_regime.empty else None,
        df_oracle.iloc[0] if not df_oracle.empty else None,
    )
    adx_val = readings["adx"]
    hurst_val = readings["hurst"]
    plus_di = readings["plus_di"]
    minus_di = readings["minus_di"]
    vol_state = readings["volatility"]
    trend_pwr = readings["trend_strength"]

    regime_absent = "no market_regime row"
    oracle_absent = "no oracle_state row"
    di_absent = "no DI reading"

    adx_gauge = _gauge_display(
        adx_val,
        lambda v: f"{v:.1f}",
        lambda v: "Strong Trend (>25)" if v >= 25 else "Low Momentum (<20)",
        regime_absent,
    )
    hurst_gauge = _gauge_display(
        hurst_val,
        lambda v: f"{v:.3f}",
        lambda v: "Persistent Trending (>0.52)" if v >= 0.52 else "Mean-Reverting (<0.48)",
        regime_absent,
    )
    vol_gauge = _gauge_display(
        vol_state,
        lambda v: f"{v:.2f}",
        lambda v: "Calibrated ATR Regime",
        oracle_absent,
    )
    trend_gauge = _gauge_display(
        trend_pwr,
        lambda v: f"{v * 100:.1f}%",
        lambda v: "High Boardroom Conviction" if v > 0.35 else "Selective Filtering",
        oracle_absent,
    )
    di_known = plus_di is not None and minus_di is not None
    spread_di = (plus_di - minus_di) if di_known else None
    di_gauge = _gauge_display(
        spread_di,
        lambda v: f"{v:+.1f}",
        lambda v: f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}" if (plus_di > 0 or minus_di > 0) else "Balanced",
        di_absent,
    )

    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    with gcol1:
        st.metric(
            label="Trend Strength (ADX)",
            value=adx_gauge["value"],
            delta=adx_gauge["delta"],
            delta_arrow=adx_gauge["arrow"],
            delta_color=adx_gauge["color"]
        )
    with gcol2:
        st.metric(
            label="DI Polarity Spread",
            value=di_gauge["value"],
            delta=di_gauge["delta"],
            delta_arrow=di_gauge["arrow"],
            delta_color=di_gauge["color"]
        )
    with gcol3:
        st.metric(
            label="Persistence (Hurst)",
            value=hurst_gauge["value"],
            delta=hurst_gauge["delta"],
            delta_arrow=hurst_gauge["arrow"],
            delta_color=hurst_gauge["color"]
        )
    with gcol4:
        st.metric(
            label="Volatility State (\u03c3)",
            value=vol_gauge["value"],
            delta=vol_gauge["delta"],
            delta_arrow=vol_gauge["arrow"],
            delta_color=vol_gauge["color"]
        )
    with gcol5:
        st.metric(
            label="Signal Readiness",
            value=trend_gauge["value"],
            delta=trend_gauge["delta"],
            delta_arrow=trend_gauge["arrow"],
            delta_color=trend_gauge["color"]
        )

    # ── P0: Quantitative Alpha Gates & Exhaustion Deck (Go-Stock Framework) ──
    st.markdown("---")
    st.markdown("#### ⚡ Quantitative Alpha Gates & Exhaustion Deck (Go-Stock Framework)")
    st.caption("Real-time mathematical filtering deployed from institutional Go-Stock quantitative architecture.")

    qg_c1, qg_c2, qg_c3 = st.columns(3)
    with qg_c1:
        st.markdown("""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 14px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <b style="color: #00e5ff; font-size: 0.92rem;">1. Momentum Divergence Gate</b>
                <span style="background: rgba(0,229,255,0.15); color: #00e5ff; font-size: 0.72rem; padding: 2px 6px; border-radius: 3px; font-weight: bold;">ACTIVE & ARMED</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.80rem; line-height: 1.45; margin: 0;">
                Calculates rolling 24-bar RSI(14) and MACD histogram swings. <b>Vetoes breakout longs</b> when price prints higher highs on deteriorating momentum (bull traps), and prevents counter-trend knife catching.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with qg_c2:
        st.markdown("""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #c084fc; border-radius: 6px; padding: 14px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <b style="color: #c084fc; font-size: 0.92rem;">2. Volume Profile Value Area (VPVR)</b>
                <span style="background: rgba(192,132,252,0.15); color: #c084fc; font-size: 0.72rem; padding: 2px 6px; border-radius: 3px; font-weight: bold;">ACTIVE & ARMED</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.80rem; line-height: 1.45; margin: 0;">
                Computes 70% Value Area (POC, VAH, VAL) over rolling 48 bars. <b>Requires 4H breakouts to clear above VAH</b> into low-liquidity discovery zones, eliminating churn inside high-volume congestion nodes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with qg_c3:
        st.markdown("""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #facc15; border-radius: 6px; padding: 14px 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <b style="color: #facc15; font-size: 0.92rem;">3. TD Sequential Exhaustion Gate</b>
                <span style="background: rgba(250,204,21,0.15); color: #facc15; font-size: 0.72rem; padding: 2px 6px; border-radius: 3px; font-weight: bold;">ACTIVE & ARMED</span>
            </div>
            <p style="color: #cbd5e1; font-size: 0.80rem; line-height: 1.45; margin: 0;">
                Tom DeMark sequential 4-bar close tracking. <b>Vetoes LONG entries on TD-9 Upward Exhaustion</b> (buying climax) and alerts on TD-9 Downward Exhaustion (selling climax) for high-conviction reversals.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🔬 Live Fleet Quantitative Gate Clearance Radar (4H Multi-Asset Scan)")

    # Fetch and render the fleet quant alpha matrix
    quant_fleet_coins = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "NEAR/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "BNB/USDT", "XRP/USDT")
    df_quant_matrix = fetch_fleet_quant_alpha_matrix(quant_fleet_coins, timeframe="4h")
    if df_quant_matrix:
        st.dataframe(pd.DataFrame(df_quant_matrix), width="stretch", hide_index=True)
    else:
        st.caption("Synchronizing quantitative alpha matrix with live exchange candles...")

    # ── Jesse-Style Bootstrap Entry Significance & Monte Carlo Sequence Risk Deck ──
    st.markdown("---")
    jv_col_title, jv_col_btn = st.columns([4, 1])
    with jv_col_title:
        st.markdown("#### 🎲 Jesse-Style Bootstrap Entry Significance & Monte Carlo Stress Suite")
        st.caption(
            "Offline statistical validation (`quant_validation.py` + `strategy_base.py`): "
            "Tests declarative entry triggers against $B=2,000$ randomized bootstrap distributions, "
            "$\\pm 0.15\\cdot\\text{ATR}$ candle-wick perturbations, and $N=5,000$ trade-sequence Monte Carlo paths."
        )
    with jv_col_btn:
        force_jv = st.button("🔄 Re-Run Validation", key="btn_rerun_quant_val", width="stretch")

    try:
        from quant_validation import generate_full_validation_report
        jv_report = generate_full_validation_report(force_refresh=bool(force_jv))
        boot_map = jv_report.get("bootstrap_significance", {})
        ens_map = jv_report.get("finrl_ensemble_selector", {}) or {}
        ens_weights = ens_map.get("strategy_weights", {}) or {}
        mc_data = jv_report.get("monte_carlo", {})

        # 0. FinRL Mahalanobis Cross-Asset Turbulence & Rolling Sharpe Ensemble Header
        turb_raw = query_system_state("finrl_turbulence_json", "")
        turb_data = {}
        if turb_raw:
            try:
                turb_data = json.loads(turb_raw) if isinstance(turb_raw, str) else turb_raw
            except Exception:
                turb_data = {}
        t_idx = num(turb_data.get("turbulence_index"), 0.0)
        t_tau90 = num(turb_data.get("turbulence_tau90"), 30.0)
        t_tau99 = num(turb_data.get("turbulence_tau99"), 230.0)
        t_active = num(turb_data.get("active_threshold"), 230.0)
        t_state = str(turb_data.get("turbulence_state", "CALM_NORMAL"))
        t_pct = num(turb_data.get("percentile_rank"), 20.0)
        t_color = "#10b981" if t_state == "CALM_NORMAL" else ("#f59e0b" if t_state == "ELEVATED_STRESS" else "#ef4444")
        ens_winner = str(ens_map.get("ensemble_winner", "SWING_4H"))
        ens_w_sharpe = num(ens_map.get("winner_sharpe"), 2.0)

        ft_c1, ft_c2 = st.columns(2)
        with ft_c1:
            st.markdown(
                f"""
                <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {t_color}; border-radius: 6px; padding: 12px 15px; margin-bottom: 10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#f8fafc; font-size:0.88rem;">🌊 FinRL Mahalanobis Cross-Asset Turbulence ($d_t$)</b>
                        <span style="background:rgba(16,185,129,0.15); color:{t_color}; font-size:0.72rem; padding:2px 8px; border-radius:4px; font-weight:700;">{t_state} (P{t_pct:.0f})</span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.80rem; margin-top:6px;">
                        Live $d_t = $ <b style="color:#f8fafc;">{t_idx:.2f}</b> vs. Active Ceiling $\\tau = $ <b style="color:#38bdf8;">{t_active:.1f}</b>
                        ($\\tau_{{90}}={t_tau90:.1f}$, $\\tau_{{99}}={t_tau99:.1f}$ across 6-asset covariance matrix)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with ft_c2:
            st.markdown(
                f"""
                <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 12px 15px; margin-bottom: 10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="color:#f8fafc; font-size:0.88rem;">🏆 FinRL Rolling Validation Sharpe Selector</b>
                        <span style="background:rgba(0,229,255,0.15); color:#00e5ff; font-size:0.72rem; padding:2px 8px; border-radius:4px; font-weight:700;">LEADER: {ens_winner} (Sharpe {ens_w_sharpe:+.2f})</span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.80rem; margin-top:6px;">
                        Dynamic Allocation Multipliers:
                        <b style="color:#00e5ff;">SWING_4H 1.20x</b> •
                        <b style="color:#f59e0b;">SCALP_SHORT 1.05x</b> •
                        <b style="color:#c084fc;">GRID_MAKER 0.90x</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 1. Signal Edge Cards (SWING_4H, GRID_MAKER, SCALP_SHORT)
        st.markdown("##### 🧪 Bootstrap Rule Significance ($B = 2,000$ Resamples & $\\pm 0.15\\cdot\\text{ATR}$ Wick Perturbation)")
        bs_cols = st.columns(3)
        strat_meta = [
            ("SWING_4H", "4H Swing Kumo Breakout", "#00e5ff"),
            ("GRID_MAKER", "5m Chop Grid Dip-Buy", "#c084fc"),
            ("SCALP_SHORT", "5m Quick Scalp Short", "#f59e0b"),
        ]
        for idx_s, (s_key, s_label, s_color) in enumerate(strat_meta):
            s_info = boot_map.get(s_key, {})
            w_info = ens_weights.get(s_key, {})
            p_val = num(s_info.get("p_value"), 0.5)
            z_val = num(s_info.get("z_score"), 0.0)
            obs_roe = num(s_info.get("observed_expectancy_roe_pct"), 0.0)
            rand_roe = num(s_info.get("random_baseline_roe_pct"), 0.0)
            k_cnt = int(num(s_info.get("Sampled Signals (K)"), 0.0))
            surv_pct = num(s_info.get("ratchet_survival_rate_pct"), 0.0)
            r_sharpe = num(s_info.get("rolling_sharpe"), w_info.get("rolling_sharpe", 0.0))
            w_mult = num(w_info.get("allocation_weight_mult"), 1.0)
            w_posture = str(w_info.get("posture", "STANDARD"))
            is_sig = bool(s_info.get("significant_edge", p_val < 0.05))
            badge_bg = "rgba(16,185,129,0.18)" if is_sig else "rgba(245,158,11,0.18)"
            badge_fg = "#10b981" if is_sig else "#f59e0b"
            badge_txt = f"PASS (p={p_val:.4f} < 0.05)" if is_sig else f"REGIME-BOUND (p={p_val:.3f})"

            with bs_cols[idx_s]:
                st.markdown(
                    f"""
                    <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {s_color}; border-radius: 6px; padding: 14px 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <b style="color: {s_color}; font-size: 0.92rem;">{s_label}</b>
                            <span style="background: {badge_bg}; color: {badge_fg}; font-size: 0.72rem; padding: 2px 7px; border-radius: 4px; font-weight: 700;">{badge_txt}</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.81rem; line-height: 1.55;">
                            • <b>Observed vs. Random ROE:</b> <span style="color:#f8fafc;">{obs_roe:+.2f}%</span> vs. <span style="color:#94a3b8;">{rand_roe:+.2f}%</span> (Z = {z_val:+.2f})<br>
                            • <b>FinRL Rolling Sharpe & Weight:</b> <span style="color:#10b981; font-weight:600;">{r_sharpe:+.2f}</span> → <span style="color:#f8fafc; font-weight:600;">{w_mult:.2f}x ({w_posture})</span><br>
                            • <b>Ratchet Survival (±0.15 ATR):</b> <span style="color:#38bdf8; font-weight:600;">{surv_pct:.1f}%</span> (K = {k_cnt})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 2. Monte Carlo Sequence Risk & Equity Cone Fan
        mc_unavailable = bool(mc_data) and mc_data.get("status") == "unavailable"
        if mc_data and not mc_unavailable:
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown(f"##### 📈 Candle-Shuffling & Sequence Monte Carlo Fan ($N = {mc_data.get('n_simulations', 5000):,}$ Simulated Paths)")
            mc_c1, mc_c2, mc_c3, mc_c4 = st.columns(4)
            mc_c1.metric(
                "Quarter-Kelly Sizing",
                f"{mc_data.get('quarter_kelly_fraction_pct', 9.0):.1f}%",
                delta=f"Payoff {mc_data.get('empirical_payoff_ratio', 1.5):.2f}x",
            )
            mc_c2.metric(
                "Median Max Drawdown (DD50)",
                f"{mc_data.get('median_max_dd_pct', 0.0):.2f}%",
                delta="Expected Normal Sequence",
            )
            mc_c3.metric(
                "95th Pct Drawdown (MaxDD_95)",
                f"{mc_data.get('max_dd_95_pct', 0.0):.2f}%",
                delta="5% Tail Sequence Risk",
            )
            mc_c4.metric(
                "99th Pct Drawdown (MaxDD_99)",
                f"{mc_data.get('max_dd_99_pct', 0.0):.2f}%",
                delta=f"Median 60-Trade Eq: ${mc_data.get('terminal_equity_p50', 500.0):,.2f}",
            )

            cone = mc_data.get("equity_cone", {})
            steps_x = cone.get("steps", [])
            if steps_x:
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Scatter(
                    x=steps_x, y=cone.get("p95", []),
                    mode="lines", line=dict(color="rgba(16,185,129,0.35)", width=1, dash="dot"),
                    name="95th Percentile (Upper Cone)"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps_x, y=cone.get("p05", []),
                    mode="lines", line=dict(color="rgba(239,68,68,0.35)", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(56,189,248,0.07)",
                    name="5th Percentile (Worst-Case Tail)"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps_x, y=cone.get("p75", []),
                    mode="lines", line=dict(color="rgba(56,189,248,0.45)", width=1),
                    name="75th Percentile"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps_x, y=cone.get("p25", []),
                    mode="lines", line=dict(color="rgba(56,189,248,0.45)", width=1),
                    fill="tonexty", fillcolor="rgba(56,189,248,0.14)",
                    name="25th Percentile (IQR Band)"
                ))
                fig_mc.add_trace(go.Scatter(
                    x=steps_x, y=cone.get("p50", []),
                    mode="lines", line=dict(color="#00e5ff", width=2.5),
                    name="50th Percentile (Median Trajectory)"
                ))
                fig_mc.update_layout(
                    height=310,
                    margin=dict(l=20, r=20, t=25, b=25),
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#0b0f19",
                    font=dict(color="#cbd5e1", size=11),
                    xaxis=dict(title="Simulated Forward Trades (Sequence Shuffling)", gridcolor="#1e293b"),
                    yaxis=dict(title="Projected Equity ($ USDT)", gridcolor="#1e293b"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_mc, width="stretch")
        elif mc_unavailable:
            st.warning(
                "Monte Carlo sequence-risk validation is unavailable — "
                f"{mc_data.get('reason', 'no recorded trades with a return')}. "
                "The figures below are not shown because they cannot be measured."
            )

        # A silent subset of the corpus is the one thing this panel used to be able to hide: the
        # report carries the arm -> reasons map, the CLI prints it, and the panel read the same
        # object and dropped the key. _missing_dataset_warnings decides what to say; this loop only
        # prints it, and an empty or absent map prints nothing.
        for _gap_msg in _missing_dataset_warnings(jv_report.get("missing_datasets")):
            st.warning(_gap_msg)
    except Exception as jv_err:
        st.caption(f"Validation telemetry note: {jv_err}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: SYSTEM HEALTH & MOBILE ACCESS (5.5 Mobile & Public URL Access)
# ═════════════════════════════════════════════════════════════════════════════
with tab_system:
    st.subheader("📱 Mobile & Remote Access Hub (5.5)")
    local_ip = get_local_ip()
    
    mob_col1, mob_col2 = st.columns(2)
    with mob_col1:
        st.markdown(f"""
        <div style='background: #0d121f; border: 1px solid #00e5ff; border-radius: 8px; padding: 16px; margin-bottom: 15px;'>
            <h4 style='color: #00e5ff; margin-top: 0;'>📶 Local Wi-Fi Access (Phone Browser)</h4>
            <p style='color: #cbd5e1; font-size: 0.9rem;'>Connect your phone to the same Wi-Fi network as this computer and navigate to:</p>
            <div style='background: #06080e; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 1.1rem; color: #00ff66; font-weight: bold;'>
                http://{local_ip}:8501
            </div>
            <p style='color: #64748b; font-size: 0.75rem; margin-top: 8px;'>Works instantly on iPhone Safari and Android Chrome without any downloads.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with mob_col2:
        st.markdown("""
        <div style='background: #0d121f; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin-bottom: 15px;'>
            <h4 style='color: #ff9900; margin-top: 0;'>🌐 Global Remote Access (Anywhere / Cellular)</h4>
            <p style='color: #cbd5e1; font-size: 0.9rem;'>To view this dashboard outside the house on mobile cellular data:</p>
            <div style='background: #06080e; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 0.85rem; color: #00e5ff;'>
                .venv\\Scripts\\python.exe start_tunnel.py
            </div>
            <p style='color: #64748b; font-size: 0.75rem; margin-top: 8px;'>Generates an instant secure Cloudflare HTTPS URL for 24/7 remote monitoring.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Autonomous Daemon Subsystems (Unified Super Bot Architecture)")
    
    daemon_list = [
        ("Master Subprocess Orchestrator", "start_system.py", "Active Background Orchestrator (Single-Instance Socket Port 49199)"),
        ("Unified Super Bot (Dual-Engine & Scalper)", "super_bot.py", "V36.6 Dual-Regime Orchestrator: 4H Swing + 5m Chop Grid + Quick Scalp Short Engine"),
        ("Quick Scalp Short Engine", "super_bot.py / swing_trader.py", "High-Velocity Scalper: +0.90% TP / +0.40% BE Ratchet / 45m Hard Time Stop / Zero Swings"),
        ("Swing Architect Breakout Engine", "swing_trader.py", "4H Ichimoku Kumo + ATR Runway + Dynamic Trailing Stops"),
        ("Quantitative Alpha Gates (Go-Stock)", "indicators.py", "Divergence Detection, 70% Value Area Profile, TD-Sequential 9/13"),
        ("Bitget UTA Execution Client", "bitget_uta_client.py", "Live Unified Account REST/WebSocket + Native Strategy Stops"),
        ("Multi-Agent Conviction Boardroom", "conviction_filter.py", "Adversarial Red/Blue Team + Macro Anchor + Leader Gate"),
        ("Continuous In-Flight Shepherd", "super_bot.py / swing_trader.py", "15m Live Position Sentinel + Breakeven Ratchet + Dynamic Floor"),
        ("Order Flow Imbalance (OFI) Daemon", "ofi_daemon.py", "L2 Order Book Multi-Depth Delta & Microstructure Stream"),
        ("Macro Sentinel & Session Blueprint", "macro_sentinel.py", "6-Hour Global Macro Synthesis + Key Level Roadmap"),
        ("Autonomous Executive Briefing", "executive_briefing.py", "5-Minute Continuous Briefings & 1-Hour Email Rollup"),
        ("Exchange Reconciliation & Sweeper", "reconciliation_engine.py", "30s State Drift Healer & Ghost Position Purger"),
        ("Mobile & Remote Access Hub", "dashboard.py / start_tunnel.py", "Streamlit HUD (Port 8501) + Cloudflare HTTPS Tunnel")
    ]
    
    st_df_daemons = pd.DataFrame(daemon_list, columns=["Subsystem Component", "Core Script", "Operational SLA"])
    st.dataframe(st_df_daemons, width="stretch", hide_index=True)

# ── End-of-run data feed health handoff ──────────────────────────────────────
# Stash this pass's failures so the next render can surface them in the top banner.
if FEED_FAILURES:
    st.session_state["_carried_feed_failures"] = dict(FEED_FAILURES)
else:
    st.session_state.pop("_carried_feed_failures", None)
