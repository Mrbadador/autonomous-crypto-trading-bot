"""
dashboard.py — Sovereign Swarm Institutional Command Center HUD (V36.2 Enterprise)
High-Contrast, Real-Time Trading, Microstructure, Calibration, Risk & Asset Breakdown Dashboard.
Includes Auto-Refresh, Dual-Layer Stop Loss, Liquidation Distance, Margin Gauge, Chronological Curves & Operator Drawer.
"""

import os
import sys
import json
import html  # L-1 FIX: HTML escaping for unsafe_allow_html=True blocks
import sqlite3
import pytz
import requests
import socket
from datetime import datetime, timezone, timedelta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import config

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign Swarm — Command HUD",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Paths & SQLite Connections ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADING_DATA_DB = os.path.join(BASE_DIR, "trading_data.db")
TRADING_LEDGER_DB = os.path.join(BASE_DIR, "trading_ledger.db")

def connect_readonly(db_path: str, timeout: float = 10.0) -> sqlite3.Connection:
    """Connects to SQLite in WAL Read-Only mode to eliminate lock contention."""
    db_uri = f"file:{os.path.abspath(db_path).replace(os.sep, '/')}?mode=ro"
    conn = sqlite3.connect(db_uri, timeout=timeout, check_same_thread=False, uri=True)
    return conn

def query_db(db_path: str, query: str, params: tuple = ()) -> pd.DataFrame:
    """Safely executes a query and returns a pandas DataFrame."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn = connect_readonly(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

def query_system_state(key: str, default: str = "") -> str:
    """Fetches a single key value from the system_state table."""
    df = query_db(TRADING_DATA_DB, "SELECT value FROM system_state WHERE key = ?", (key,))
    if not df.empty and df.iloc[0]["value"] is not None:
        return str(df.iloc[0]["value"])
    return default

def execute_safe_write(db_path: str, query: str, params: tuple = ()) -> bool:
    """Executes a non-blocking parameterized write to SQLite with timeout."""
    try:
        with sqlite3.connect(db_path, timeout=5.0) as conn:
            conn.execute(query, params)
            conn.commit()
        return True
    except Exception:
        return False

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


def _win_rate_value(win_rate_pct, closed_trade_count) -> str:
    """Renders the fleet win rate, or says that it was never measured.

    Same convention as the main dashboard's KPI card: a rate over closed trades needs a
    population, so a fleet that closed none has no rate to publish. The old code published
    72.9% (and 0.0%) whether or not anything had been measured.
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


def _profit_factor_delta(profit_factor, win_count, loss_count) -> str:
    """Renders the fleet profit factor, or says that it was never measured.

    A profit factor needs a denominator: with no losing trade there is no ratio to publish, so
    the old 2.62 / 1.0 constants are gone and the absence names its cause instead.
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


def _calibration_panel_values(c_metrics) -> dict:
    """Renders the calibration panel's three metrics, or says that none was measured.

    Same convention as the main dashboard's calibration panel: the tracker returns
    total_samples=0 with NO overall_win_rate and NO calibration_grade for an empty calibration
    log (and a brier_score of 0.0 standing in for a score with no samples to average over), so
    the `.get(key, 0)` defaults used to publish 0%, 0.0000 and a GOOD grade nobody earned.
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
        signals_text = "0 Trades"
        absent_reason = "no signals evaluated yet"
    else:
        signals_text = f"{int(samples)} Trades"
        absent_reason = "not measured"

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
    state_file = os.path.join(BASE_DIR, "microstructure_state.json")
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

    # 2. Batch Binance API (fallback / global coverage)
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=1.0)
        if resp.status_code == 200:
            for item in resp.json():
                prices[item["symbol"]] = float(item["price"])
    except Exception:
        pass
    return prices

# ── High-Contrast Cyberpunk UI Styling ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&family=Roboto+Mono:wght@400;700&display=swap');

:root {
    --bg-main: #06080e;
    --card-bg: #0d121f;
    --neon-cyan: #00e5ff;
    --neon-green: #00ff66;
    --neon-danger: #ff003c;
    --neon-orange: #ff9900;
    --border-color: #1e293b;
}

.stApp {
    background-color: var(--bg-main);
    color: #e2e8f0;
    font-family: 'Outfit', sans-serif;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-left: 3px solid var(--neon-cyan) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Roboto Mono', 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] p {
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #94a3b8 !important;
    font-weight: 700 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    color: #94a3b8 !important;
    text-transform: uppercase;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: rgba(0, 229, 255, 0.08) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    color: #00e5ff !important;
}

/* Badges & Tables */
.badge-green { background: rgba(0,255,102,0.15); color: #00ff66; border: 1px solid #00ff66; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
.badge-red { background: rgba(255,0,60,0.15); color: #ff003c; border: 1px solid #ff003c; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
.badge-neutral { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid #94a3b8; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
.badge-cyan { background: rgba(0,229,255,0.15); color: #00e5ff; border: 1px solid #00e5ff; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }
.badge-orange { background: rgba(255,153,0,0.15); color: #ff9900; border: 1px solid #ff9900; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-family: monospace; }

.pos-card {
    background: #0d121f;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls & Configuration ─────────────────────────────────────────
with st.sidebar:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=40)
    st.title("⚡ Sovereign Swarm")
    st.caption("Institutional Autonomous Multi-Agent Trading Fleet")
    st.markdown("---")
    
    # P0: Auto-Refresh Toggle Engine
    st.markdown("##### ⏱️ Telemetry Refresh Engine")
    auto_refresh_enabled = st.checkbox("⚡ Live Auto-Refresh (4s)", value=True, help="Refreshes HUD telemetry automatically every 4 seconds.")
    if auto_refresh_enabled and AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=4000, key="sovereign_swarm_hud_autorefresh")
    
    st.markdown("---")

    # P1: Operator Safe Controls Drawer
    with st.expander("🛠️ Fleet Operator Controls", expanded=False):
        st.markdown("##### ⚙️ Operational Overrides")
        curr_op_halt = query_system_state("OPERATOR_HALT", "0") in ("1", "true", "TRUE")
        pause_entries = st.toggle(
            "⏸️ Pause Entries (Close-Only)",
            value=curr_op_halt,
            help="Sticky manual pause (`OPERATOR_HALT`). Halts all new trade entries across the fleet while letting existing positions trail safely to SL/TP."
        )
        if pause_entries != curr_op_halt:
            new_val = "1" if pause_entries else "0"
            execute_safe_write(TRADING_DATA_DB, "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('OPERATOR_HALT', ?, datetime('now'))", (new_val,))
            if execute_safe_write(TRADING_DATA_DB, "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('RISK_HALT', ?, datetime('now'))", (new_val,)):
                st.toast(f"Fleet State Updated: {'Entries Paused (Close-Only)' if pause_entries else 'Manual Pause Cleared'}")
                st.rerun()

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Force Exchange Re-Sync", use_container_width=True, help="Dispatches an immediate re-sync signal to the Reconciliation Engine."):
            if execute_safe_write(TRADING_DATA_DB, "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('force_resync', '1', datetime('now'))", ()):
                st.toast("✅ Force Re-Sync signal dispatched to Reconciliation Engine!")

    st.markdown("---")
    # A missing balance key means the account's equity was never read. The panel says so instead
    # of publishing a demo balance and a profit that was never measured (the main dashboard reads
    # the same key the same way).
    _balance_raw = str(query_system_state("balance", "") or "").strip()
    try:
        live_bal_val = float(_balance_raw) if _balance_raw else None
    except (TypeError, ValueError):
        live_bal_val = None
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
    # This app never had a third tranche, so there is none to record.
    capital = _capital_display(total_deposited, deposit_t1, deposit_t2, None)
    if live_bal_val is not None and total_deposited is not None:
        total_profit = live_bal_val - total_deposited
        total_roi_pct = (total_profit / total_deposited) * 100.0 if total_deposited > 0 else 0.0
    else:
        total_profit = None
        total_roi_pct = None

    st.write(f"**Connected Exchange:** Bitget UTA (Live)")
    if live_bal_val is not None:
        st.write(f"**Current Equity:** `${live_bal_val:.2f} USDT`")
    else:
        st.write("**Current Equity:** `⚠️ Telemetry Offline (Syncing...)`")
    st.write(f"**Deposited Capital:** `{capital['deposited_usdt']}`")
    st.caption(f"↳ Seed: `{capital['seed']}` | Top-up: `{capital['topup1']}`")
    if total_profit is None:
        st.markdown(f"**Real Net Profit:** <span style='font-weight:700;'>{_profit_display(total_profit, total_roi_pct)}</span>", unsafe_allow_html=True)
    else:
        profit_sign = "+$" if total_profit >= 0 else "-$"
        profit_color = "#00ff66" if total_profit >= 0 else "#ff003c"
        st.markdown(f"**Real Net Profit:** <span style='color:{profit_color}; font-weight:700;'>{profit_sign}{abs(total_profit):.2f} USDT ({total_roi_pct:+.2f}%)</span>", unsafe_allow_html=True)
    local_ip = get_local_ip()
    st.write(f"**Local Mobile URL:** `http://{local_ip}:8501`")
    st.markdown("---")
    st.caption(f"Last HUD Refresh: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")

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

# ── Dynamic Account Sizing & Performance Calculations ────────────────────────
# Fetch All Closed Trades from all fleet trade tables
df_t1 = query_db(TRADING_DATA_DB, "SELECT * FROM trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t2 = query_db(TRADING_DATA_DB, "SELECT * FROM berserker_trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t3 = query_db(TRADING_DATA_DB, "SELECT * FROM marl_live_trades WHERE status = 'closed' OR status = 'CLOSED'")
df_t4 = query_db(TRADING_DATA_DB, "SELECT * FROM mean_reversion_trades WHERE status = 'closed' OR status = 'CLOSED'")
trade_dfs = [d for d in [df_t1, df_t2, df_t3, df_t4] if not d.empty]
df_all_trades = pd.concat(trade_dfs, ignore_index=True) if trade_dfs else pd.DataFrame()

if not df_all_trades.empty and 'timestamp' in df_all_trades.columns:
    df_all_trades = df_all_trades.sort_values(by='timestamp', ascending=False).reset_index(drop=True)

total_closed_count = len(df_all_trades)
if not df_all_trades.empty:
    wins_df = df_all_trades[df_all_trades['pnl'].astype(float) > 0]
    losses_df = df_all_trades[df_all_trades['pnl'].astype(float) <= 0]
    win_count = len(wins_df)
    loss_count = len(losses_df)
    win_rate_pct = (win_count / total_closed_count) * 100.0 if total_closed_count > 0 else None
    gross_profit = wins_df['pnl'].astype(float).sum()
    gross_loss = abs(losses_df['pnl'].astype(float).sum())
    # No losing trade means no denominator, so there is no ratio to publish: the card prints the
    # absence and its cause (see _profit_factor_delta) instead of a constant that was never
    # measured.
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    net_realized_pnl = gross_profit - gross_loss
else:
    # Nothing was measured: no trades, and every rate and total an absence. This branch used to
    # publish a demo ledger (43 wins, 16 losses, 72.9%, 2.62x, $12.18) as if it were telemetry.
    win_count, loss_count, win_rate_pct, profit_factor, net_realized_pnl = 0, 0, None, None, None

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
total_margin_locked = 0.0
total_unrealized_pnl = 0.0
live_prices = fetch_live_market_prices()

if not df_open.empty:
    for _, row in df_open.iterrows():
        s = str(row['symbol']).upper().replace("/", "").replace("-", "")
        active_coins_set.add(s)
        qty = float(row.get('qty') or 0.0)
        ep = float(row.get('entry_price') or 0.0)
        lev = float(row.get('leverage') or (10.0 if "BTC" in s else 5.0))
        curr_p = float(live_prices.get(s, ep) or ep)
        side = str(row.get('side') or 'long').lower()
        
        pos_margin = (qty * ep) / lev if lev > 0 else (qty * ep)
        total_margin_locked += pos_margin
        pos_unr = (curr_p - ep) * qty if side == 'long' else (ep - curr_p) * qty
        total_unrealized_pnl += pos_unr

max_coin_slots = int(query_system_state("GLOBAL_MAX_POSITIONS", "5"))
margin_limit_pct = float(query_system_state("GLOBAL_MARGIN_LIMIT", "70.0"))
bal_for_calc = live_bal_val if live_bal_val is not None else 0.0
free_cash_buffer = max(0.0, bal_for_calc - total_margin_locked)

# Margin utilization calculation
margin_util_pct = (total_margin_locked / bal_for_calc * 100.0) if bal_for_calc > 0 else 0.0
margin_bar_width = min(100.0, (margin_util_pct / margin_limit_pct) * 100.0)
# The buffer and the utilisation are ratios against the account balance. With no balance there is
# no denominator, so they are absences rather than zeroes.
margin_measured = live_bal_val is not None and live_bal_val > 0
if not margin_measured:
    margin_color = "#64748b"
    margin_status = "UNKNOWN (no account balance)"
elif margin_util_pct < 45.0:
    margin_color = "#00ff66"
    margin_status = "SAFE BUFFER"
elif margin_util_pct < 65.0:
    margin_color = "#ff9900"
    margin_status = "ELEVATED EXPOSURE"
else:
    margin_color = "#ff003c"
    margin_status = "CRITICAL / NEARING CEILING"
margin_util_display = f"{margin_util_pct:.1f}%" if margin_measured else "n/a (no account balance)"
free_cash_display = f"${free_cash_buffer:.2f} USDT" if margin_measured else "n/a (no account balance)"

# ── Hero KPI Cards Grid ──────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="💰 Live Account Equity",
        value=f"${live_bal_val:.2f} USDT" if live_bal_val is not None else "Syncing...",
        delta=(f"{'+$' if total_profit>=0 else '-$'}{abs(total_profit):.2f} ({total_roi_pct:+.2f}% ROI)"
               if total_profit is not None else None),
    )

with kpi2:
    st.metric(
        label="🏆 Fleet Win Rate & PF",
        value=_win_rate_value(win_rate_pct, total_closed_count),
        delta=_profit_factor_delta(profit_factor, win_count, loss_count),
        # An absent ratio is not a fall from the last reading, so it must not wear the green
        # up-arrow st.metric gives every delta that does not start with "-".
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
    st.metric(
        label="⚡ Live Unrealized PnL",
        value=f"{'+$' if total_unrealized_pnl>=0 else '-$'}{abs(total_unrealized_pnl):.2f} USDT",
        delta=(f"${free_cash_buffer:.2f} Free Liquid Cash" if margin_measured
               else "Free Liquid Cash not measured (no account balance)")
    )

# P0: Global Margin Utilization Visual Progress Bar
st.markdown(f"""
<div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <span style="font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">
            🛡️ Portfolio Margin Utilization: <b style="color: {margin_color};">{margin_util_display}</b> of {margin_limit_pct:.0f}% Hard Ceiling ({margin_status})
        </span>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #e2e8f0;">
            Margin Locked: <b style="color: {margin_color};">${total_margin_locked:.2f}</b> | Liquid Buffer: <b style="color: #00e5ff;">{free_cash_display}</b>
        </span>
    </div>
    <div style="width: 100%; background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; position: relative;">
        <div style="width: {margin_bar_width:.1f}%; background: {margin_color}; height: 100%; border-radius: 5px; transition: width 0.4s ease-in-out;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main Tabbed Navigation Deck ──────────────────────────────────────────────
tab_live, tab_ledger, tab_ai, tab_risk, tab_system = st.tabs([
    "🎯 LIVE FLEET & RADAR",
    "📜 CLOSED TRADES & LEDGER",
    "🧠 AI BOARDROOM & MEMORY",
    "🛡️ RISK & MICROSTRUCTURE",
    "⚙️ SYSTEM HEALTH & DAEMONS"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE FLEET & WATCHLIST RADAR (Dual-Layer Stops + Liquidation + Age)
# ═════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.subheader("🎯 Active Live Positions (Real-Time Price & Risk Radar)")
    
    if not df_open.empty:
        # Render visual metric cards for each live position
        pos_cols = st.columns(len(df_open))
        display_pos = []
        
        for idx, row in df_open.iterrows():
            sym = str(row['symbol'])
            sym_clean = sym.upper().replace("/", "").replace("-", "")
            side = str(row.get('side') or 'long').lower()
            qty = float(row.get('qty') or 0.0)
            entry_p = float(row.get('entry_price') or 0.0)
            sl = float(row.get('stop_loss') or 0.0)
            tp = float(row.get('take_profit') or 0.0)
            hard_sl = float(row.get('hard_sl') or 0.0)
            lev = float(row.get('leverage') or (10.0 if "BTC" in sym_clean else 5.0))
            margin = (qty * entry_p) / lev if lev > 0 else (qty * entry_p)
            curr_p = float(live_prices.get(sym_clean, entry_p) or entry_p)
            entry_time_raw = row.get('entry_time')
            pos_age = compute_position_age(entry_time_raw)
            sl_breach_raw = row.get('sl_breach_time')
            has_flash_breach = bool(sl_breach_raw and str(sl_breach_raw).strip() not in ("", "None", "0", "N/A"))

            # Fallback Hard SL if not explicitly set in database
            if hard_sl <= 0.0 and entry_p > 0:
                hard_sl = entry_p * (1.0 - 0.035) if side == 'long' else entry_p * (1.0 + 0.035)

            # Liquidation estimation
            est_liq = calculate_estimated_liquidation(entry_p, lev, side)
            dist_liq_pct = calculate_liq_distance_pct(curr_p, est_liq, side)

            # Unrealized PnL & ROI
            unr_pnl = (curr_p - entry_p) * qty if side == 'long' else (entry_p - curr_p) * qty
            unr_roi = (unr_pnl / margin * 100.0) if margin > 0 else 0.0
            
            # Target Distances
            if side == "long":
                dist_tp_pct = ((tp - curr_p) / curr_p * 100.0) if tp > 0 else 0.0
                dist_sl_pct = ((curr_p - sl) / curr_p * 100.0) if sl > 0 else 0.0
                tp_str = f"+{dist_tp_pct:.2f}% to TP" if tp > 0 else "Active Trail"
                sl_str = f"{dist_sl_pct:.2f}% above SL" if sl > 0 else "Active Trail"
                is_risk_free = (sl >= entry_p)
            else:
                dist_tp_pct = ((curr_p - tp) / curr_p * 100.0) if tp > 0 else 0.0
                dist_sl_pct = ((sl - curr_p) / curr_p * 100.0) if sl > 0 else 0.0
                tp_str = f"-{dist_tp_pct:.2f}% to TP" if tp > 0 else "Active Trail"
                sl_str = f"{dist_sl_pct:.2f}% below SL" if sl > 0 else "Active Trail"
                is_risk_free = (sl <= entry_p and sl > 0)

            with pos_cols[idx % len(pos_cols)]:
                pnl_color = "#00ff66" if unr_pnl >= 0 else "#ff003c"
                risk_free_badge = "<span style='background: rgba(0,255,102,0.15); color: #00ff66; border: 1px solid #00ff66; font-size: 0.68rem; padding: 1px 5px; border-radius: 3px; font-weight: bold;'>🛡️ RISK-FREE</span>" if is_risk_free else ""
                flash_timer_badge = "<div style='margin-top: 4px; background: rgba(255,0,60,0.2); border: 1px solid #ff003c; border-radius: 4px; padding: 2px 6px; font-size: 0.72rem; color: #ff003c; font-weight: bold;'>⚠️ FLASH CRASH TIMER RUNNING</div>" if has_flash_breach else ""

                card_html = (
                    f'<div class="pos-card" style="border-left: 3px solid {pnl_color};">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div><b style="font-size: 1.1rem; color: #ffffff;">{sym}</b> {risk_free_badge}</div>'
                    f'<span class="{"badge-green" if side=="long" else "badge-red"}">{side.upper()} {lev:.0f}x</span>'
                    f'</div>'
                    f'<div style="margin: 8px 0; font-family: monospace; font-size: 1.25rem; font-weight: bold; color: {pnl_color};">'
                    f'{"+$" if unr_pnl>=0 else "-$"}{abs(unr_pnl):.4f} ({unr_roi:+.2f}%)'
                    f'</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">'
                    f'Entry: <b>${entry_p:,.4f}</b> | Mark: <b style="color: #00e5ff;">${curr_p:,.4f}</b><br>'
                    f'⏱️ Age: <b style="color: #e2e8f0;">{pos_age}</b> | Locked: <b style="color: #e2e8f0;">${margin:.2f}</b><br>'
                    f'🎯 Target: <b style="color: #00ff66;">{tp_str} (${tp:,.4f})</b><br>'
                    f'🛡️ Soft SL: <b style="color: #ff9900;">${sl:,.4f} ({sl_str})</b><br>'
                    f'🧱 Hard SL: <b style="color: #94a3b8;">${hard_sl:,.4f} (Exchange)</b><br>'
                    f'☠️ Est. Liq: <b style="color: #ff003c;">${est_liq:,.4f} (+{dist_liq_pct:.1f}% safety)</b>'
                    f'{flash_timer_badge}'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)
            
            display_pos.append({
                "Asset": sym,
                "Side": side.upper(),
                "Leverage": f"{lev:.0f}x",
                "Size": f"{qty:.4f}",
                "Entry Price": f"${entry_p:,.4f}",
                "Live Price": f"${curr_p:,.4f}",
                "Unrealized PnL": f"{'+$' if unr_pnl>=0 else '-$'}{abs(unr_pnl):.4f}",
                "ROI (%)": f"{unr_roi:+.2f}%",
                "Soft SL": f"${sl:,.4f}",
                "Hard SL (Bitget)": f"${hard_sl:,.4f}",
                "Est. Liq Price": f"${est_liq:,.4f}",
                "Liq Safety Buffer": f"+{dist_liq_pct:.1f}%",
                "Age": pos_age,
                "Margin Locked": f"${margin:.2f}",
                "Status": "🟢 RISK-FREE" if is_risk_free else "🟢 LIVE EXECUTED"
            })
            
        st.markdown("#### 📋 Detailed Active Position & Risk Matrix")
        st.dataframe(pd.DataFrame(display_pos), width="stretch", hide_index=True)
    else:
        st.info("No active open positions. The fleet is currently sidelined in cash awaiting high-conviction breakout signals.")

    st.markdown("---")
    st.subheader("📡 Top 10 Watchlist Radar (Live Technical Screening)")
    
    raw_allowed = query_system_state("ALLOWED_SYMBOLS")
    if raw_allowed:
        try:
            top_coins = json.loads(raw_allowed)
        except Exception:
            top_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]
    else:
        top_coins = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "BNB/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "LINK/USDT", "NEAR/USDT"]

    radar_data = []
    for coin in top_coins:
        coin_clean = coin.replace("/", "").replace("-", "")
        is_active = any(coin_clean in c for c in active_coins_set)
        c_price = live_prices.get(coin_clean, 0.0)
        is_core = coin in ["BTC/USDT", "ETH/USDT"]
        coin_lev = config.get_asset_leverage(coin) if hasattr(config, "get_asset_leverage") else (10 if "BTC" in coin else 5)
        
        radar_data.append({
            "Coin": coin,
            "Live Price": f"${c_price:,.4f}" if c_price > 0 else "---",
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
            df_filtered = df_filtered.sort_values(by='timestamp', ascending=False)
        elif f_sort == "Oldest First":
            df_filtered = df_filtered.sort_values(by='timestamp', ascending=True)
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
        df_ledger_display['PnL ($)'] = df_ledger_display['pnl'].apply(lambda x: f"{'+$' if float(x)>=0 else '-$'}{abs(float(x)):.4f}")
        df_ledger_display['Result'] = df_ledger_display['pnl'].apply(lambda x: "🟢 WIN" if float(x)>0 else "🔴 LOSS")
        df_ledger_display['Entry Price'] = df_ledger_display['entry_price'].apply(lambda x: f"${float(x):,.4f}" if x else "$0.00")
        df_ledger_display['Exit Price'] = df_ledger_display['exit_price'].apply(lambda x: f"${float(x):,.4f}" if x else "$0.00")
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
            show_coin_df['Win Rate (%)'] = show_coin_df['win_rate'].apply(lambda x: f"{x:.1f}%")
            show_coin_df['Net PnL ($)'] = show_coin_df['net_pnl'].apply(lambda x: f"{'+$' if x>=0 else '-$'}{abs(x):.2f}")
            show_coin_df['Avg / Trade'] = show_coin_df['avg_pnl'].apply(lambda x: f"{'+$' if x>=0 else '-$'}{abs(x):.2f}")
            show_coin_df = show_coin_df[['symbol', 'trades', 'Win Rate (%)', 'Net PnL ($)', 'Avg / Trade']].rename(columns={'symbol': 'Asset', 'trades': 'Trades'})
            st.dataframe(show_coin_df, width="stretch", hide_index=True)

        # ── 5.4 Trade Duration Analysis ───────────────────────────────────────
        st.markdown("---")
        st.subheader("⏱️ Trade Duration & Holding Time Analysis")
        
        dur1, dur2, dur3, dur4 = st.columns(4)
        # These four cards used to publish hardcoded demo figures (38.5 mins, 46.2 mins, 27.8 mins,
        # a 1.66x hold ratio) that no trade was ever measured against. This view does not carry a
        # duration column, so there is no holding-time measurement to publish: the cards say so.
        with dur1:
            st.metric("Avg Fleet Duration", "n/a (not measured)",
                      delta="this view does not measure holding time")
        with dur2:
            st.metric("Avg Winner Duration", "n/a (not measured)",
                      delta="this view does not measure holding time")
        with dur3:
            st.metric("Avg Loser Duration", "n/a (not measured)",
                      delta="this view does not measure holding time")
        with dur4:
            st.metric("Hold Ratio (Win/Loss)", "n/a (not measured)",
                      delta="this view does not measure holding time")

        st.info("💡 **Holding Asymmetry Insight:** Not measured in this view — the closed-trade ledger carries no duration column here, so the winner/loser holding ratio is not computed and no asymmetry is claimed.")
    else:
        st.info("No closed trades recorded in the database yet.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: AI BOARDROOM & MEMORY
# ═════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🧠 Multi-Agent AI Boardroom & Real-Time Telemetry")
    
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        st.markdown("#### 📡 Recent AI Boardroom Decision Audits")
        df_telem = query_db(TRADING_DATA_DB, "SELECT timestamp, symbol, action, price, vol_pct, quality, reasoning FROM telemetry ORDER BY timestamp DESC LIMIT 8")
        if not df_telem.empty:
            df_telem['Time (SAST)'] = df_telem['timestamp'].apply(utc_to_sast)
            df_telem['Price'] = df_telem['price'].apply(lambda x: f"${float(x):,.2f}" if x else "---")
            df_telem['Vol Ratio'] = df_telem['vol_pct'].apply(lambda x: f"{float(x or 0):.1f}%")
            show_telem = df_telem[['Time (SAST)', 'symbol', 'action', 'Price', 'Vol Ratio', 'reasoning']].rename(columns={'symbol': 'Asset', 'action': 'Action', 'reasoning': 'AI Rationale'})
            st.dataframe(show_telem, width="stretch", hide_index=True)
        else:
            st.info("Telemetry queue synchronizing...")

    with col_ai2:
        st.markdown("#### 📚 Active Dynamic Rules Learned from Past Mistakes")
        df_rules = query_db(TRADING_LEDGER_DB, "SELECT category, rule_text, confidence, created_at FROM learned_rules ORDER BY rowid DESC LIMIT 6")
        if not df_rules.empty:
            for _, r in df_rules.iterrows():
                st.markdown(f"""
                <div style='background: #0d121f; border-left: 3px solid #00e5ff; padding: 8px 12px; margin-bottom: 8px; border-radius: 4px; font-size: 0.85rem;'>
                    <b style='color: #00e5ff;'>[{r['category']}] ({float(r['confidence'])*100:.0f}% Conf)</b><br>
                    <span style='color: #e2e8f0;'>{r['rule_text']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            # The rule count used to be a hardcoded "33": nothing had counted anything.
            df_rule_total = query_db(TRADING_LEDGER_DB, "SELECT COUNT(*) as cnt FROM learned_rules")
            if df_rule_total.empty:
                st.info("Learned-rule count unavailable (ledger unreadable).")
            else:
                st.info(f"{int(df_rule_total.iloc[0]['cnt'])} dynamic neural rules active in long-term memory.")

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
    # The lesson count used to be a hardcoded "220": nothing had counted anything.
    df_lessons_total = query_db(TRADING_LEDGER_DB, "SELECT COUNT(*) as cnt FROM trade_lessons")
    if df_lessons_total.empty:
        st.subheader("📖 Post-Trade Forensic Lessons Extracted")
    else:
        total_lessons_db = int(df_lessons_total.iloc[0]['cnt'])
        st.subheader(f"📖 Post-Trade Forensic Lessons Extracted ({total_lessons_db} Lessons in Database)")
    df_lessons = query_db(TRADING_LEDGER_DB, "SELECT symbol, pnl, lesson, timestamp FROM trade_lessons ORDER BY id DESC LIMIT 5")
    if not df_lessons.empty:
        for _, l in df_lessons.iterrows():
            pnl_v = float(l['pnl'] or 0)
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
        st.write(f"• **Dynamic Leverage:** 3.0x CHOP / 5.0x TREND *(Auto-Adapting)*")
        st.write(f"• **Core Allocation:** 2 Reserved Slots for BTC/ETH")
        st.write(f"• **Altcoin Ceiling:** 3 Coins Max Concurrent (out of {max_coin_slots})")
        st.write(f"• **Global Margin Limit:** {margin_limit_pct:.1f}% Ceiling (30% Cash Buffer)")
        st.write(f"• **Max Entries per Coin:** 2 Positions Max")

    with rcol2:
        st.markdown("#### 🌊 Macro & News Event Gating")
        df_macro = query_db(TRADING_LEDGER_DB, "SELECT * FROM macro_market_context ORDER BY id DESC LIMIT 1")
        if not df_macro.empty:
            spy_p = float(df_macro.iloc[0].get('spy_price', 0.0) or 0.0)
            uup_p = float(df_macro.iloc[0].get('uup_price', 0.0) or 0.0)
            dxy_p = float(df_macro.iloc[0].get('dxy_price', 0.0) or 0.0)
            st.write(f"• **S&P 500 (SPY):** ${spy_p:,.2f}")
            st.write(f"• **USD Index (DXY):** {dxy_p:.2f}")
            st.write(f"• **Invesco DB US Dollar (UUP):** ${uup_p:.2f}")
            st.write(f"• **Macro Regime:** {df_macro.iloc[0].get('macro_regime', 'NEUTRAL')}")
        st.write(f"• **News Calendar Gate:** ±30m window around FOMC/CPI/NFP")
        st.write(f"• **Hurst Exponent Filter:** Hurst >= 0.52 for Trend Breakouts")

    with rcol3:
        st.markdown("#### 🔬 Microstructure Veto Pulse")
        last_sniper_veto = query_system_state("last_veto_sniper", "None")
        market_state = _published_state(query_system_state("market_regime_label", ""), "no market regime label published")
        warden_state = _published_state(query_system_state("warden_60m_regime", ""), "no 60m warden regime published")

        reg_color = _stance_color(market_state["text"], market_state["measured"],
                                  ("BULL_TREND", "BULLISH", "TRENDING"), ("BEAR_TREND", "BEARISH"))
        warden_color = _stance_color(warden_state["text"], warden_state["measured"],
                                     ("LONG_ONLY",), ("SHORT_ONLY",))
        # L-1 FIX: Escape all DB/LLM-derived text before interpolation into unsafe_allow_html=True
        safe_market_label = html.escape(market_state["text"])
        safe_warden_regime = html.escape(warden_state["text"])
        safe_last_veto = html.escape(str(last_sniper_veto))
        market_note = "" if market_state["measured"] else _absence_note(market_state["reason"])
        warden_note = "" if warden_state["measured"] else _absence_note(warden_state["reason"])
        st.write(f"• **Market Regime Label:** <b style='color: {reg_color};'>{safe_market_label}</b>{market_note}", unsafe_allow_html=True)
        st.write(f"• **60m Warden Bias:** <b style='color: {warden_color};'>{safe_warden_regime}</b>{warden_note}", unsafe_allow_html=True)
        st.write(f"• **Volatility Cascade Risk:** <b style='color: #94a3b8;'>not measured</b> <span style='color: #94a3b8; font-size: 0.8rem;'>(no cascade telemetry is published)</span>", unsafe_allow_html=True)
        veto_preview = safe_last_veto[:40] + ("..." if len(safe_last_veto) > 40 else "")
        st.write(f"• **Last Filter Gate:** <span style='color: #94a3b8; font-size: 0.8rem;'>{veto_preview}</span>", unsafe_allow_html=True)

    # P1: Visual OFI Depth Imbalance Meter & Regime Strength Deck
    st.markdown("---")
    st.markdown("#### 🔬 Real-Time Order Flow Imbalance (OFI) & Regime Strength Deck")

    df_micro = query_db(TRADING_LEDGER_DB, "SELECT symbol, liquidity_imbalance, bid_wall_price, ask_wall_price, updated_at FROM oracle_microstructure ORDER BY id DESC LIMIT 1")
    imb = 0.0
    bid_wall = 0.0
    ask_wall = 0.0
    micro_sym = "BTC/USDT"
    if not df_micro.empty:
        imb = float(df_micro.iloc[0].get('liquidity_imbalance', 0.0) or 0.0)
        bid_wall = float(df_micro.iloc[0].get('bid_wall_price', 0.0) or 0.0)
        ask_wall = float(df_micro.iloc[0].get('ask_wall_price', 0.0) or 0.0)
        micro_sym = str(df_micro.iloc[0].get('symbol', 'BTC/USDT'))

    # Visual bipolar OFI bar calculation (-1.0 to +1.0 mapped to 0% to 100%)
    marker_pct = max(0.0, min(100.0, ((imb + 1.0) / 2.0) * 100.0))
    imb_color = "#00ff66" if imb > 0.05 else ("#ff003c" if imb < -0.05 else "#94a3b8")
    bias_label = "AGGRESSIVE BUY PRESSURE" if imb > 0.10 else ("AGGRESSIVE SELL PRESSURE" if imb < -0.10 else "BALANCED LIQUIDITY")

    st.markdown(f"""
    <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
            <span style="color: #ff003c; font-weight: 700;">🔴 SELL PRESSURE (-1.0)</span>
            <span style="color: {imb_color}; font-weight: 800; font-size: 1.05rem;">
                {micro_sym} OFI: {imb:+.3f} [{bias_label}]
            </span>
            <span style="color: #00ff66; font-weight: 700;">🟢 BUY PRESSURE (+1.0)</span>
        </div>
        <div style="position: relative; width: 100%; height: 16px; background: linear-gradient(90deg, rgba(255,0,60,0.6) 0%, rgba(30,41,59,0.7) 50%, rgba(0,255,102,0.6) 100%); border-radius: 8px;">
            <div style="position: absolute; left: {marker_pct}%; top: -4px; width: 8px; height: 24px; background: #ffffff; border: 1px solid #000; border-radius: 4px; transform: translateX(-50%); box-shadow: 0 0 10px #ffffff;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
            <span>🟢 Major Bid Wall: <b style="color: #00ff66;">${bid_wall:,.2f}</b></span>
            <span>⚡ Stream: <b style="color: #00e5ff;">100ms L2 Order Book</b></span>
            <span>🔴 Major Ask Wall: <b style="color: #ff003c;">${ask_wall:,.2f}</b></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quantitative Regime Deck Gauges (ADX + Hurst Exponent)
    df_regime = query_db(TRADING_DATA_DB, "SELECT * FROM market_regime ORDER BY id DESC LIMIT 1")
    df_oracle = query_db(TRADING_LEDGER_DB, "SELECT * FROM oracle_state ORDER BY id DESC LIMIT 1")

    readings = _regime_readings(
        df_regime.iloc[0] if not df_regime.empty else None,
        df_oracle.iloc[0] if not df_oracle.empty else None,
    )
    adx_val = readings["adx"]
    hurst_val = readings["hurst"]
    vol_state = readings["volatility"]
    trend_pwr = readings["trend_strength"]

    regime_absent = "no market_regime row"
    oracle_absent = "no oracle_state row"

    adx_gauge = _gauge_display(
        adx_val,
        lambda v: f"{v:.1f}",
        lambda v: "Strong Trend (>25)" if v >= 25 else "Low Momentum Chop (<20)",
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

    gcol1, gcol2, gcol3, gcol4 = st.columns(4)
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
            label="Persistence (Hurst)",
            value=hurst_gauge["value"],
            delta=hurst_gauge["delta"],
            delta_arrow=hurst_gauge["arrow"],
            delta_color=hurst_gauge["color"]
        )
    with gcol3:
        st.metric(
            label="Volatility State (\u03c3)",
            value=vol_gauge["value"],
            delta=vol_gauge["delta"],
            delta_arrow=vol_gauge["arrow"],
            delta_color=vol_gauge["color"]
        )
    with gcol4:
        st.metric(
            label="Signal Readiness",
            value=trend_gauge["value"],
            delta=trend_gauge["delta"],
            delta_arrow=trend_gauge["arrow"],
            delta_color=trend_gauge["color"]
        )

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
        ("Master Coordinator & Launcher", "start_system.py", "Active Background Task (task-6230)"),
        ("Live Execution & Reconciler", "main.py / live_paper_exchange.py", "Bitget UTA Live Pilot Mode"),
        ("Berserker Momentum Engine", "berserker_trader.py", "5m/15m Breakout Scalper"),
        ("MARL Reinforcement Engine", "marl_live_trader.py", "Microstructure Q-Policy"),
        ("Portfolio Risk Sentinel", "portfolio_risk_agent.py", "70% Margin Ceiling Guardian"),
        ("Reconciliation & Profit Sweeper", "reconciliation_engine.py", "30s Bitget Auto-Healer & Sweeper"),
        ("Macro Trend Warden", "warden_60m.py", "4H/1D Directional Trend Lock"),
        ("Macro Economic Calendar Gate", "economic_calendar.py", "±30m Event Suppressor"),
        ("Live Microstructure Oracle", "live_oracle.py", "Order Book Imbalance Engine"),
        ("Email Dispatcher & Notifier", "email_dispatcher.py", "Real-Time Google TLS SMTP Alerts"),
        ("Mobile Remote Access Bridge", "start_tunnel.py", "Wi-Fi & Cloudflare Secure Gateway")
    ]
    
    st_df_daemons = pd.DataFrame(daemon_list, columns=["Subsystem", "Module", "Operational Role"])
    st_df_daemons = st_df_daemons.rename(columns={"Subsystem": "Subsystem Component", "Module": "Core Script", "Operational Role": "Operational SLA"})
    st.dataframe(st_df_daemons, width="stretch", hide_index=True)
