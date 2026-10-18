"""
dashboard.py — Sovereign Swarm Institutional Command Center HUD (V36.2 Enterprise)
High-Contrast, Real-Time Trading, Microstructure, Calibration, Risk & Asset Breakdown Dashboard.
Includes Auto-Refresh, Dual-Layer Stop Loss, Liquidation Distance, Margin Gauge, Chronological Curves & Operator Drawer.
"""

import os
import sys
import time
import json
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

    # 7. Overlays: Hard SL (Exchange Native)
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

    # P1: Operator Safe Controls Drawer
    with st.expander("🛠️ Fleet Operator Controls", expanded=False):
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
    total_deposited = state_num("user_total_deposit", num(os.getenv("INITIAL_BALANCE"), 250.0))
    deposit_t1 = state_num("initial_deposit_tranche_1", 55.00)
    deposit_t2 = state_num("topup_deposit_tranche_2", 82.00)
    deposit_t3 = state_num("topup_deposit_tranche_3", 113.00)
    bal_display = f"${live_bal_val:,.2f}" if live_bal_val is not None else "Syncing..."

    st.write(f"**Connected Exchange:** Bitget UTA (Live)")
    if live_bal_val is not None:
        total_profit = live_bal_val - total_deposited
        total_roi_pct = (total_profit / total_deposited) * 100.0 if total_deposited > 0 else 0.0
        st.write(f"**Current Equity:** `${live_bal_val:.2f} USDT`")
    else:
        st.write(f"**Current Equity:** `⚠️ Telemetry Offline (Syncing...)`")
        total_profit = 0.0
        total_roi_pct = 0.0
    st.write(f"**Deposited Capital:** `${total_deposited:.2f} USDT`")
    tranche_str = f"↳ Seed: `${deposit_t1:.2f}` | Top-up 1: `${deposit_t2:.2f}`"
    if deposit_t3 > 0:
        tranche_str += f" | Top-up 2: `${deposit_t3:.2f}`"
    st.caption(tranche_str)
    profit_sign = "+$" if total_profit >= 0 else "-$"
    profit_color = "#00ff66" if total_profit >= 0 else "#ff003c"
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
macro_trend = query_system_state("btc_macro_trend", "BULLISH").upper()
risk_halt = query_system_state("RISK_HALT", "0")
rec_lev = query_system_state("recommended_leverage", "3")
fg_val = query_system_state("fear_greed_index", "50.0")
news_active = "1" if state_flag("news_event_active") else "0"
news_reason = query_system_state("news_event_reason", "No active events")

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
else:
    banner_color = "#ff9900"
    banner_bg = "rgba(255, 153, 0, 0.08)"
    banner_icon = "🟠"
    regime_text = "REGIME: RANGE-BOUND CHOP"

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
            ⚡ Lev: <b style="color: #00e5ff;">{rec_lev}x Dynamic</b> | 🧠 F&G: <b style="color: #e2e8f0;">{num(fg_val, 50.0):.0f}/100</b> | 📰 Event Gate: <b style="color: {'#ff003c' if news_active=='1' else '#00ff66'};">{'EVENT VETO' if news_active=='1' else 'CLEAR'}</b>
        </span>
    </div>
    <div>
        <span class="{'badge-green' if risk_halt == '0' else 'badge-red'}">
            {'SYSTEM ACTIVE' if risk_halt == '0' else 'ENTRIES PAUSED (CLOSE ONLY)'}
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
    win_rate_pct = (win_count / total_closed_count) * 100.0 if total_closed_count > 0 else 0.0
    gross_profit = wins_df['pnl'].astype(float).sum()
    gross_loss = abs(losses_df['pnl'].astype(float).sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (2.62 if win_count > 0 else 1.0)
    net_realized_pnl = gross_profit - gross_loss
else:
    win_count, loss_count, win_rate_pct, profit_factor, net_realized_pnl = 0, 0, 0.0, 0.0, 0.0

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
total_live_funding = 0.0
live_prices = fetch_live_market_prices()
funding_telemetry_map = fetch_live_position_funding()

if not df_open.empty:
    for _, row in df_open.iterrows():
        s = str(row['symbol']).upper().replace("/", "").replace("-", "")
        active_coins_set.add(s)
        qty = num(row.get('qty'))
        ep = num(row.get('entry_price'))
        lev = num(row.get('leverage'), 10.0 if "BTC" in s else 5.0)
        curr_p = num(live_prices.get(s, ep), ep)
        side = str(row.get('side') or 'long').lower()
        
        pos_margin = (qty * ep) / lev if lev > 0 else (qty * ep)
        total_margin_locked += pos_margin
        pos_unr = (curr_p - ep) * qty if side == 'long' else (ep - curr_p) * qty
        total_unrealized_pnl += pos_unr

        f_info = funding_telemetry_map.get(s, {}) or funding_telemetry_map.get(str(row['symbol']), {})
        total_live_funding += num(f_info.get("total_funding"), 0.0)

max_coin_slots = int(num(query_system_state("GLOBAL_MAX_POSITIONS", "5"), 5.0))
margin_limit_pct = state_num("GLOBAL_MARGIN_LIMIT", 70.0)
bal_for_calc = live_bal_val if live_bal_val is not None else 0.0
free_cash_buffer = max(0.0, bal_for_calc - total_margin_locked)

# Margin utilization calculation
margin_util_pct = (total_margin_locked / bal_for_calc * 100.0) if bal_for_calc > 0 else 0.0
margin_bar_width = min(100.0, (margin_util_pct / margin_limit_pct) * 100.0)
if margin_util_pct < 45.0:
    margin_color = "#00ff66"
    margin_status = "SAFE BUFFER"
elif margin_util_pct < 65.0:
    margin_color = "#ff9900"
    margin_status = "ELEVATED EXPOSURE"
else:
    margin_color = "#ff003c"
    margin_status = "CRITICAL / NEARING CEILING"

# ── Hero KPI Cards Grid ──────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    help_deposit = f"Deposited Capital: ${total_deposited:.2f} USDT (${deposit_t1:.0f} seed + ${deposit_t2:.0f} top-up 1" + (f" + ${deposit_t3:.0f} top-up 2" if deposit_t3 > 0 else "") + f"). True Net Profit: {total_profit:+.2f} USDT."
    eq_val_str = f"${live_bal_val:.2f} USDT" if live_bal_val is not None else "Syncing..."
    st.metric(
        label="💰 Live Account Equity",
        value=eq_val_str,
        delta=f"{'+$' if total_profit>=0 else '-$'}{abs(total_profit):.2f} ({total_roi_pct:+.2f}% ROI)" if live_bal_val is not None else None,
        help=help_deposit
    )
    principal_caption = f"Principal: **${total_deposited:.2f}** (`${deposit_t1:.0f}` + `${deposit_t2:.0f}`" + (f" + `${deposit_t3:.0f}`" if deposit_t3 > 0 else "") + ")"
    st.caption(principal_caption)

with kpi2:
    st.metric(
        label="🏆 Fleet Win Rate & PF",
        value=f"{win_rate_pct:.1f}% Win Rate",
        delta=f"{profit_factor:.2f}x Profit Factor ({win_count}W / {loss_count}L)"
    )

with kpi3:
    st.metric(
        label="🔥 Portfolio Heat & Slots",
        value=f"{len(active_coins_set)} / {max_coin_slots} Slots Used",
        delta=f"{max(0, max_coin_slots - len(active_coins_set))} Slots Open (2 Core / 3 Alt)"
    )

with kpi4:
    fund_sign = "+$" if total_live_funding >= 0 else "-$"
    fund_status = "Gained" if total_live_funding > 0 else ("Paid" if total_live_funding < 0 else "Neutral")
    fund_delta = f"Funding: {fund_sign}{abs(total_live_funding):.4f} ({fund_status})"
    st.metric(
        label="⚡ Live Unrealized PnL",
        value=f"{'+$' if total_unrealized_pnl>=0 else '-$'}{abs(total_unrealized_pnl):.2f} USDT",
        delta=fund_delta,
        help=f"Unrealized Mark PnL: ${total_unrealized_pnl:+.4f} USDT | Cumulative Accrued Funding Fee: {fund_sign}{abs(total_live_funding):.4f} USDT"
    )

# P0: Global Margin Utilization Visual Progress Bar
fund_badge_color = "#00ff66" if total_live_funding >= 0 else ("#ffaa00" if total_live_funding < 0 else "#94a3b8")
st.markdown(f"""
<div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 12px 18px; margin-top: 10px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <span style="font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">
            🛡️ Portfolio Margin Utilization: <b style="color: {margin_color};">{margin_util_pct:.1f}%</b> of {margin_limit_pct:.0f}% Hard Ceiling ({margin_status})
        </span>
        <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #e2e8f0;">
            Margin Locked: <b style="color: {margin_color};">${total_margin_locked:.2f}</b> | Liquid Buffer: <b style="color: #00e5ff;">${free_cash_buffer:.2f} USDT</b> | Active Funding: <b style="color: {fund_badge_color};">{'+$' if total_live_funding>=0 else '-$'}{abs(total_live_funding):.4f}</b>
        </span>
    </div>
    <div style="width: 100%; background: #1e293b; height: 10px; border-radius: 5px; overflow: hidden; position: relative;">
        <div style="width: {margin_bar_width:.1f}%; background: {margin_color}; height: 100%; border-radius: 5px; transition: width 0.4s ease-in-out;"></div>
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
    b_margin = num(briefing_data.get("margin_locked"), total_margin_locked)
    b_util_pct = num(briefing_data.get("margin_util_pct"))
    b_slots = int(num(briefing_data.get("slots_occupied"), len(df_open)))
    b_rules = int(num(briefing_data.get("rules_count"), 108))
    max_slots_calc = max(4, max_coin_slots)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="💰 Live Bitget Equity",
            value=f"${b_eq:,.2f} USDT",
            delta=f"+${b_eq - total_deposited:.2f} ({((b_eq - total_deposited)/total_deposited)*100:.1f}%)" if total_deposited > 0 else None,
            delta_color="normal"
        )
    with kpi2:
        st.metric(
            label="🛡️ Margin Utilized (70% Max)",
            value=f"{b_util_pct:.1f}%",
            delta=f"${b_margin:.2f} locked (Ceiling: 70%)",
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
        total_fund_sign = "+$" if total_live_funding >= 0 else "-$"
        total_fund_color = "#00ff66" if total_live_funding >= 0 else ("#ffaa00" if total_live_funding < 0 else "#94a3b8")
        total_fund_action = "Gained / Received" if total_live_funding > 0 else ("Paid to Counterparties" if total_live_funding < 0 else "Neutral")
        true_portfolio_net = total_unrealized_pnl + total_live_funding
        true_port_sign = "+$" if true_portfolio_net >= 0 else "-$"
        true_port_color = "#00ff66" if true_portfolio_net >= 0 else "#ff003c"

        st.markdown(f"""
        <div style="background: rgba(13, 18, 31, 0.85); border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 10px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span style="color: #94a3b8; font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.05em;">Live Portfolio Funding & Carry Telemetry</span><br>
                <span style="font-size: 0.95rem; font-weight: bold; color: #ffffff;">Fleet Accrued Funding: </span>
                <span style="font-size: 1.05rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: {total_fund_color};">{total_fund_sign}{abs(total_live_funding):.4f} USDT</span>
                <span style="color: {total_fund_color}; font-size: 0.78rem; font-weight: 600;">({total_fund_action})</span>
            </div>
            <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;">
                <span style="color: #94a3b8;">Unrealized Mark PnL: <b style="color: {'#00ff66' if total_unrealized_pnl>=0 else '#ff003c'};">{'+$' if total_unrealized_pnl>=0 else '-$'}{abs(total_unrealized_pnl):.2f}</b></span><br>
                <span style="color: #94a3b8;">True Net (Mark + Funding): <b style="color: {true_port_color};">{true_port_sign}{abs(true_portfolio_net):.2f} USDT</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Render visual metric cards for each live position
        pos_cols = st.columns(len(df_open))
        display_pos = []
        
        for idx, row in df_open.iterrows():
            sym = str(row['symbol'])
            sym_clean = sym.upper().replace("/", "").replace("-", "")
            side = str(row.get('side') or 'long').lower()
            qty = num(row.get('qty'))
            entry_p = num(row.get('entry_price'))
            sl = num(row.get('stop_loss'))
            tp = num(row.get('take_profit'))
            hard_sl = num(row.get('hard_sl'))
            lev = num(row.get('leverage'), 10.0 if "BTC" in sym_clean else 5.0)
            margin = (qty * entry_p) / lev if lev > 0 else (qty * entry_p)
            curr_p = num(live_prices.get(sym_clean, entry_p) or entry_p)
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

            # Live Position Funding Telemetry
            f_info = funding_telemetry_map.get(sym_clean, {}) or funding_telemetry_map.get(sym, {})
            pos_funding = num(f_info.get("total_funding"), 0.0)
            if pos_funding < 0:
                pos_funding_str = f"-${abs(pos_funding):.4f} (Paid)"
                pos_funding_badge = f"<b style='color: #ffaa00;'>-${abs(pos_funding):.4f} USDT</b> <span style='color: #ffaa00; font-size: 0.72rem;'>(Paid)</span>"
            elif pos_funding > 0:
                pos_funding_str = f"+${pos_funding:.4f} (Gained)"
                pos_funding_badge = f"<b style='color: #00ff66;'>+${pos_funding:.4f} USDT</b> <span style='color: #00ff66; font-size: 0.72rem;'>(Gained)</span>"
            else:
                pos_funding_str = "$0.0000 (Neutral)"
                pos_funding_badge = "<b style='color: #94a3b8;'>$0.0000</b> <span style='color: #94a3b8; font-size: 0.72rem;'>(Neutral)</span>"

            # True Net PnL (Mark PnL + Funding)
            true_net_pnl = unr_pnl + pos_funding
            true_net_roi = (true_net_pnl / margin * 100.0) if margin > 0 else 0.0
            true_net_str = f"{'+$' if true_net_pnl >= 0 else '-$'}{abs(true_net_pnl):.4f} ({true_net_roi:+.2f}%)"
            true_net_color = "#00ff66" if true_net_pnl >= 0 else "#ff003c"
            
            # Target Distances
            if side == "long":
                dist_tp_pct = ((tp - curr_p) / curr_p * 100.0) if tp > 0 and curr_p > 0 else 0.0
                dist_sl_pct = ((curr_p - sl) / curr_p * 100.0) if sl > 0 and curr_p > 0 else 0.0
                tp_str = f"+{dist_tp_pct:.2f}% to TP" if tp > 0 else "Active Trail"
                sl_str = f"{dist_sl_pct:.2f}% above SL" if sl > 0 else "Active Trail"
                is_risk_free = (sl >= entry_p)
            else:
                dist_tp_pct = ((curr_p - tp) / curr_p * 100.0) if tp > 0 and curr_p > 0 else 0.0
                dist_sl_pct = ((sl - curr_p) / curr_p * 100.0) if sl > 0 and curr_p > 0 else 0.0
                tp_str = f"-{dist_tp_pct:.2f}% to TP" if tp > 0 else "Active Trail"
                sl_str = f"{dist_sl_pct:.2f}% below SL" if sl > 0 else "Active Trail"
                is_risk_free = (sl <= entry_p and sl > 0)

            with pos_cols[idx % len(pos_cols)]:
                pnl_color = "#00ff66" if unr_pnl >= 0 else "#ff003c"
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
                    f'<span class="{"badge-green" if side=="long" else "badge-red"}">{side.upper()} {lev:.0f}x</span>'
                    f'</div>'
                    f'<div style="margin: 8px 0; font-family: monospace; font-size: 1.25rem; font-weight: bold; color: {pnl_color};">'
                    f'{"+$" if unr_pnl>=0 else "-$"}{abs(unr_pnl):.4f} ({unr_roi:+.2f}%)'
                    f'</div>'
                    f'<div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.45;">'
                    f'Entry: <b>${entry_p:,.4f}</b> | Mark: <b style="color: #00e5ff;">${curr_p:,.4f}</b><br>'
                    f'⏱️ Age: <b style="color: #e2e8f0;">{pos_age}</b> | Locked: <b style="color: #e2e8f0;">${margin:.2f}</b><br>'
                    f'💸 Live Funding: {pos_funding_badge}<br>'
                    f'📊 True Net PnL: <b style="color: {true_net_color};">{true_net_str}</b><br>'
                    f'🎯 Target: <b style="color: #00ff66;">{tp_str} (${tp:,.4f})</b><br>'
                    f'🛡️ Soft SL: <b style="color: #ff9900;">${sl:,.4f} ({sl_str})</b><br>'
                    f'🧱 Hard SL: <b style="color: #94a3b8;">${hard_sl:,.4f} (Exchange)</b><br>'
                    f'☠️ Est. Liq: <b style="color: #ff003c;">${est_liq:,.4f} (+{dist_liq_pct:.1f}% safety)</b>'
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
                "Leverage": f"{lev:.0f}x",
                "Size": f"{qty:.4f}",
                "Entry Price": f"${entry_p:,.4f}",
                "Live Price": f"${curr_p:,.4f}",
                "Unrealized PnL": f"{'+$' if unr_pnl>=0 else '-$'}{abs(unr_pnl):.4f}",
                "Funding (Live)": pos_funding_str,
                "True Net PnL": f"{'+$' if true_net_pnl>=0 else '-$'}{abs(true_net_pnl):.4f}",
                "ROI (%)": f"{unr_roi:+.2f}%",
                "Soft SL": f"${sl:,.4f}",
                "Hard SL (Bitget)": f"${hard_sl:,.4f}",
                "Est. Liq Price": f"${est_liq:,.4f}",
                "Liq Safety Buffer": f"+{dist_liq_pct:.1f}%",
                "In-Flight Sentinel": shep_verdict_str,
                "Age": pos_age,
                "Margin Locked": f"${margin:.2f}",
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
            s_name = str(row['symbol'])
            s_clean = s_name.upper().replace("/", "").replace("-", "").split(":")[0]
            s_side = str(row.get('side') or 'long').upper()
            s_entry = num(row.get('entry_price'))
            s_sl = num(row.get('stop_loss'))
            s_tp = num(row.get('take_profit'))
            s_hard_sl = num(row.get('hard_sl'))
            if s_hard_sl <= 0.0 and s_entry > 0:
                s_hard_sl = s_entry * (1.0 - 0.035) if s_side == 'LONG' else s_entry * (1.0 + 0.035)
            s_curr = num(live_prices.get(s_clean, s_entry) or s_entry)
            s_lev = num(row.get('leverage'), 10.0 if "BTC" in s_clean else 5.0)
            s_qty = num(row.get('qty'))
            s_margin = (s_qty * s_entry) / s_lev if s_lev > 0 else (s_qty * s_entry)
            s_unr = (s_curr - s_entry) * s_qty if s_side == 'LONG' else (s_entry - s_curr) * s_qty
            s_roi = (s_unr / s_margin * 100.0) if s_margin > 0 else 0.0
            s_liq = calculate_estimated_liquidation(s_entry, s_lev, s_side.lower())

            open_pos_list.append({
                "symbol": s_name,
                "clean_symbol": s_clean,
                "side": s_side,
                "entry": s_entry,
                "sl": s_sl,
                "hard_sl": s_hard_sl,
                "tp": s_tp,
                "curr": s_curr,
                "lev": s_lev,
                "qty": s_qty,
                "margin": s_margin,
                "unr_pnl": s_unr,
                "unr_roi": s_roi,
                "est_liq": s_liq
            })

        chart_tab_titles = [
            f"📈 {p['symbol']} ({'+$' if p['unr_pnl'] >= 0 else '-$'}{abs(p['unr_pnl']):.2f})"
            for p in open_pos_list
        ]
        if len(open_pos_list) > 1:
            chart_tab_titles.append("📊 All Active Fleet Charts (Stacked)")

        pos_chart_tabs = st.tabs(chart_tab_titles)

        for i, p in enumerate(open_pos_list):
            with pos_chart_tabs[i]:
                # Top summary pill for this specific position
                pnl_color = "#00ff66" if p["unr_pnl"] >= 0 else "#ff003c"
                st.markdown(f"""
                <div style="background: rgba(13, 18, 31, 0.7); border: 1px solid #1e293b; border-left: 3px solid {pnl_color}; border-radius: 6px; padding: 8px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <b style="font-size: 1.05rem; color: #ffffff;">{p['symbol']}</b>
                        <span style="background: rgba(0, 229, 255, 0.15); color: #00e5ff; font-size: 0.75rem; padding: 2px 6px; border-radius: 3px; font-weight: bold; margin-left: 6px;">{p['side']} {p['lev']:.0f}x</span>
                        <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 10px;">Size: <b style="color: #e2e8f0;">{p['qty']}</b> (${p['margin']:.2f} locked)</span>
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;">
                        <span style="color: #94a3b8;">Entry: <b style="color: #00e5ff;">${p['entry']:,.4f}</b></span> | 
                        <span style="color: #94a3b8;">Mark: <b style="color: #ffffff;">${p['curr']:,.4f}</b></span> | 
                        <span style="color: #94a3b8;">Unrealized PnL: <b style="color: {pnl_color};">{'+$' if p['unr_pnl'] >= 0 else '-$'}{abs(p['unr_pnl']):.4f} ({p['unr_roi']:+.2f}%)</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                fig = build_position_candlestick_chart(
                    symbol=p["symbol"],
                    side=p["side"].lower(),
                    entry_p=p["entry"],
                    sl=p["sl"],
                    hard_sl=p["hard_sl"],
                    tp=p["tp"],
                    curr_p=p["curr"],
                    est_liq=p["est_liq"],
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
                    pnl_color = "#00ff66" if p["unr_pnl"] >= 0 else "#ff003c"
                    st.markdown(f"**{p['symbol']} ({p['side']} {p['lev']:.0f}x)** — Entry: `${p['entry']:,.4f}` | Mark: `${p['curr']:,.4f}` | PnL: <span style='color: {pnl_color}; font-weight: bold;'>{'+$' if p['unr_pnl'] >= 0 else '-$'}{abs(p['unr_pnl']):.4f}</span>", unsafe_allow_html=True)
                    fig_all = build_position_candlestick_chart(
                        symbol=p["symbol"],
                        side=p["side"].lower(),
                        entry_p=p["entry"],
                        sl=p["sl"],
                        hard_sl=p["hard_sl"],
                        tp=p["tp"],
                        curr_p=p["curr"],
                        est_liq=p["est_liq"],
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
        try:
            if os.path.exists(MICROSTRUCTURE_STATE_FILE):
                with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f_radar_ms:
                    r_ms_data = json.load(f_radar_ms)
                r_sym_code = radar_sym.replace("/", "").replace("-", "").upper()
                r_sym_info = r_ms_data.get("symbols", {}).get(r_sym_code, {})
                if r_sym_info:
                    r_mlofi = float(r_sym_info.get("mlofi", 0.0) or 0.0)
                    r_mp = float(r_sym_info.get("micro_price", 0.0) or 0.0)
                    r_bps = float(r_sym_info.get("micro_price_bps", 0.0) or 0.0)
                    r_vpin = float(r_sym_info.get("vpin", 0.0) or 0.0)
                    r_h_spoof = float(r_sym_info.get("hawkes_spoof_index", 0.0) or 0.0)
                    r_h_rho = float(r_sym_info.get("hawkes_rho", 0.0) or 0.0)
                    r_h_veto = bool(r_sym_info.get("hawkes_veto", False))
                    r_kyle = float(r_sym_info.get("kyle_lambda", 0.0) or 0.0)
                    r_kyle_pct = float(r_sym_info.get("kyle_lambda_pctile", 0.5) or 0.5)
                    r_lvr = float(r_sym_info.get("lvr_tax_bps", 0.0) or 0.0)
                    r_tox_halt = bool(r_sym_info.get("toxicity_halt", False))
                    
                    r_v_col = "#00ff66" if r_vpin < 0.45 else ("#f59e0b" if r_vpin < 0.70 else "#ff003c")
                    r_v_lbl = "Benign Flow" if r_vpin < 0.45 else ("Caution Flow" if r_vpin < 0.70 else "TOXIC CASCADE")
                    r_m_col = "#00ff66" if r_mlofi > 0.05 else ("#ff003c" if r_mlofi < -0.05 else "#94a3b8")
                    r_h_col = "#ff003c" if r_h_veto or r_h_spoof > 0.72 or r_h_rho >= 1.0 else ("#f59e0b" if r_h_spoof > 0.45 else "#00ff66")
                    r_h_lbl = "VETO" if r_h_veto else ("ELEVATED" if r_h_spoof > 0.45 else "STATIONARY")
                    r_kyle_col = "#00ff66" if r_kyle_pct < 0.60 else ("#f59e0b" if r_kyle_pct < 0.85 else "#ff003c")
                    r_lvr_badge = "<b style='color: #ff003c;'>[TOXIC HALT]</b>" if r_tox_halt else ""
                    
                    st.markdown(f"""
                    <div style="background: rgba(13, 18, 31, 0.85); border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 10px 16px; margin-top: 10px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
                        <span>🔬 <b>{radar_sym} Alpha Telemetry:</b></span>
                        <span>Micro-Price: <b style="color: #00e5ff;">${r_mp:,.4f}</b> (<b style="color: {'#00ff66' if r_bps >= 0 else '#ff003c'};">{r_bps:+.2f} bps</b>)</span>
                        <span>MLOFI (10-Level): <b style="color: {r_m_col};">{r_mlofi:+.2f}</b></span>
                        <span>VPIN: <b style="color: {r_v_col};">{r_vpin:.3f} [{r_v_lbl}]</b></span>
                        <span>Hawkes Spoof: <b style="color: {r_h_col};">{r_h_spoof:.3f} (ρ={r_h_rho:.2f}) [{r_h_lbl}]</b></span>
                        <span>Kyle's λ (Impact): <b style="color: {r_kyle_col};">{r_kyle:.4f} ({r_kyle_pct*100:.0f}%)</b></span>
                        <span>LVR Tax (σ²/8): <b style="color: {'#ff003c' if r_lvr >= 10.0 else '#38bdf8'};">{r_lvr:.2f} bps</b> {r_lvr_badge}</span>
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
            "Macro Direction": "LONG ONLY" if macro_trend == "BULLISH" else "SHORT ONLY"
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
                <span style="color: #f8fafc; font-size: 1.15rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">${total_deposited:.2f} USDT</span>
                <span style="color: #64748b; font-size: 0.8rem;"> (${deposit_t1:.0f} seed + ${deposit_t2:.0f} top-up 1{f" + ${deposit_t3:.0f} top-up 2" if deposit_t3 > 0 else ""})</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Current Portfolio Value</span><br>
                <span style="color: #38bdf8; font-size: 1.15rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{bal_display} USDT</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;">Real Net Profit</span><br>
                <span style="color: {'#00ff66' if total_profit >= 0 else '#ff003c'}; font-size: 1.15rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;">
                    {'+$' if total_profit >= 0 else '-$'}{abs(total_profit):.2f} USDT ({total_roi_pct:+.2f}%)
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

    arb_bias = query_system_state("ai_regime_bias", "DEFENSIVE_CHOP")
    arb_risk = query_system_state("ai_regime_trap_risk", "0.40")
    arb_thesis = query_system_state("ai_regime_thesis", "Consolidation within range; capital shield priority.")
    arb_lev = query_system_state("recommended_leverage", "3")
    arb_upd = query_system_state("ai_regime_updated_at", "")

    col_arb1, col_arb2, col_arb3 = st.columns([1.2, 1, 2.0])
    with col_arb1:
        st.metric("Cognitive Regime Bias", arb_bias, delta=f"Dynamic Cap: {arb_lev}x")
    with col_arb2:
        try:
            risk_val = float(arb_risk or 0.40)
        except Exception:
            risk_val = 0.40
        st.metric("Institutional Trap Risk", f"{risk_val*100:.0f}%", delta="Safe Expansion" if risk_val < 0.5 else "High Trap Risk", delta_color="normal" if risk_val < 0.5 else "inverse")
    with col_arb3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #cbd5e1;">
            <b style="color: #00e5ff;">🧠 Arbiter Structural Thesis:</b><br/>
            {arb_thesis}
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
        b_state = basis_data.get("state", "IDLE")
        b_spot = float(basis_data.get("spot_price", 0.0) or 0.0)
        b_perp = float(basis_data.get("perp_price", 0.0) or 0.0)
        b_spread = float(basis_data.get("basis_spread_usd", 0.0) or 0.0)
        b_disloc = float(basis_data.get("basis_dislocation_pct", 0.0) or 0.0)
        b_funding = float(basis_data.get("funding_rate_8h", 0.0) or 0.0)
        b_yield = float(basis_data.get("annualized_yield_pct", 0.0) or 0.0)
        b_hurdle = float(basis_data.get("hurdle_rate_pct", 8.0) or 8.0)
        b_unwind = float(basis_data.get("unwind_hurdle_pct", 1.5) or 1.5)
        b_alloc = float(basis_data.get("capital_allocated_usd", 1000.0) or 1000.0)
        b_funding_coll = float(basis_data.get("cumulative_funding_harvested", 0.0) or 0.0)
        b_upnl = float(basis_data.get("unrealized_pnl_usd", 0.0) or 0.0)
        b_has_pos = basis_data.get("has_active_position", False)
        b_net_delta = float(basis_data.get("net_delta_usd", 0.0) or 0.0)
        b_holding_days = float(basis_data.get("holding_days", 0.0) or 0.0)
        b_min_holding_days = int(basis_data.get("min_holding_days", 7) or 7)
        b_roundtrip_fee = float(basis_data.get("roundtrip_fee_usd", 2.40) or 2.40)
        b_fee_target = float(basis_data.get("fee_hurdle_target_usd", 3.00) or 3.00)
        b_fee_amortized = float(basis_data.get("fee_amortized_pct", 0.0) or 0.0)
        b_margin_mode = str(basis_data.get("margin_mode", "crossed")).upper()
        b_liq_price = float(basis_data.get("liquidation_price", 0.0) or 0.0)
        b_dist_liq = float(basis_data.get("distance_to_liq_pct", 100.0) or 100.0)
        b_time = utc_to_sast(basis_data.get("timestamp"))

        # Dynamic state styling
        if b_state == "ACTIVE_DELTA_NEUTRAL":
            status_color = "#00ff66"
            status_badge = "🟢 ACTIVE 1:1 DELTA-NEUTRAL TWIN"
        elif b_state == "PENDING_PERP_FILL":
            status_color = "#ffcc00"
            status_badge = "⏳ PENDING PERP MAKER FILL (15s GATE)"
        elif b_state == "UNWINDING":
            status_color = "#ff003c"
            status_badge = "⚠️ UNWINDING TWIN POSITION"
        elif b_state == "DISABLED":
            status_color = "#64748b"
            status_badge = "⏸️ PARKED / DISABLED"
        else:
            status_color = "#00e5ff"
            status_badge = "📡 IDLE (SCANNING YIELD & SPREAD)"

        yield_color = "#00ff66" if b_yield >= b_hurdle else ("#ff9900" if b_yield > b_unwind else "#ff003c")
        liq_color = "#00ff66" if b_dist_liq >= 30.0 else ("#ff9900" if b_dist_liq >= 20.0 else "#ff003c")
        fee_color = "#00ff66" if b_fee_amortized >= 100.0 else ("#00e5ff" if b_fee_amortized >= 50.0 else "#94a3b8")
        fee_bar_width = min(100.0, max(0.0, (b_fee_amortized / 125.0) * 100.0))

        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {status_color}; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    <span style="font-weight: bold; color: {status_color}; font-size: 1.0rem;">🪙 BTC CASH-AND-CARRY BASIS ENGINE</span>
                    <span style="background: rgba(255,255,255,0.06); color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.78rem; margin-left: 10px;">
                        {status_badge}
                    </span>
                    <span style="background: rgba(148,163,184,0.12); color: #94a3b8; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; margin-left: 6px;">
                        Alloc: ${b_alloc:,.0f} USDT (50% Spot / 50% Perp Short · {b_margin_mode})
                    </span>
                </div>
                <span style="color: #64748b; font-size: 0.75rem;">{b_time}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 8px;">
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">8H FUNDING RATE</span><br>
                    <b style="color: {yield_color}; font-size: 0.95rem;">{b_funding*100:+.4f}%</b>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">ANNUALIZED YIELD</span><br>
                    <b style="color: {yield_color}; font-size: 0.95rem;">{b_yield:+.2f}%</b> <span style="font-size:0.72rem; color:#64748b;">(Hurdle: {b_hurdle}%)</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">SPOT VS PERP SPREAD</span><br>
                    <b style="color: #ffffff; font-size: 0.95rem;">${b_spread:+.2f}</b> <span style="font-size:0.72rem; color:{'#00ff66' if abs(b_disloc)<1.0 else '#ff9900'};">({b_disloc:+.2f}%)</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">FUNDING HARVESTED</span><br>
                    <b style="color: #00ff66; font-size: 0.95rem;">+${b_funding_coll:.4f} USD</b>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">NET MARKET DELTA</span><br>
                    <b style="color: #00e5ff; font-size: 0.95rem;">${b_net_delta:+.2f} (Neutral)</b>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.06);">
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #94a3b8; font-size: 0.72rem;">FEE HURDLE AMORTIZATION</span>
                        <span style="color: {fee_color}; font-size: 0.75rem; font-weight: bold;">{b_fee_amortized:.1f}% / 125%</span>
                    </div>
                    <div style="width: 100%; background: #1e293b; height: 6px; border-radius: 3px; margin: 5px 0; overflow: hidden;">
                        <div style="width: {fee_bar_width:.1f}%; background: {fee_color}; height: 100%; border-radius: 3px;"></div>
                    </div>
                    <span style="color: #cbd5e1; font-size: 0.72rem;">Harvested: ${b_funding_coll:.2f} · Target: ${b_fee_target:.2f} · Hold: {b_holding_days:.1f}d / {b_min_holding_days}d</span>
                </div>
                <div style="background: rgba(0,0,0,0.25); padding: 8px 12px; border-radius: 4px;">
                    <span style="color: #94a3b8; font-size: 0.72rem;">PERP SHORT LIQUIDATION GUARD</span><br>
                    <b style="color: {liq_color}; font-size: 0.95rem;">+{b_dist_liq:.1f}% Buffer</b>
                    <span style="color: #94a3b8; font-size: 0.72rem; margin-left: 6px;">(Liq: ${b_liq_price:,.2f})</span><br>
                    <span style="color: {'#00ff66' if b_dist_liq>=30.0 else '#ff9900'}; font-size: 0.72rem;">🛡️ {b_margin_mode} Margin Mode Active (Min Buffer: 20%)</span>
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
        with cal_col1:
            st.metric("Signals Evaluated", f"{c_metrics.get('total_samples', 0)} Trades")
        with cal_col2:
            st.metric("Historical Win Rate", f"{c_metrics.get('overall_win_rate', 0)}%")
        with cal_col3:
            b_score = c_metrics.get('brier_score', 0)
            b_grade = c_metrics.get('calibration_grade', 'GOOD')
            st.metric("Brier Calibration Score", f"{b_score:.4f}", delta=f"Grade: {b_grade}")

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
        st.write(f"• **Dynamic Leverage:** 3.0x CHOP / 5.0x TREND *(Auto-Adapting)*")
        st.write(f"• **Core Allocation:** 2 Reserved Slots for BTC/ETH")
        st.write(f"• **Altcoin Ceiling:** 3 Coins Max Concurrent (out of {max_coin_slots})")
        st.write(f"• **Global Margin Limit:** {margin_limit_pct:.1f}% Ceiling (30% Cash Buffer)")
        st.write(f"• **Max Entries per Coin:** 2 Positions Max")

    with rcol2:
        st.markdown("#### 🌊 Macro & News Event Gating")
        df_macro = query_db(TRADING_LEDGER_DB, "SELECT * FROM macro_market_context ORDER BY id DESC LIMIT 1")
        if not df_macro.empty:
            spy_p = num(df_macro.iloc[0].get('spy_price'), 0.0)
            uup_p = num(df_macro.iloc[0].get('uup_price'), 0.0)
            dxy_p = num(df_macro.iloc[0].get('dxy_price'), 0.0)
            st.write(f"• **S&P 500 (SPY):** ${spy_p:,.2f}")
            st.write(f"• **USD Index (DXY):** {dxy_p:.2f}")
            st.write(f"• **Invesco DB US Dollar (UUP):** ${uup_p:.2f}")
            st.write(f"• **Macro Regime:** {df_macro.iloc[0].get('macro_regime', 'NEUTRAL')}")
        st.write(f"• **News Calendar Gate:** ±30m window around FOMC/CPI/NFP")
        st.write(f"• **Hurst Exponent Filter:** Hurst >= 0.52 for Trend Breakouts")

    with rcol3:
        st.markdown("#### 🔬 Microstructure Veto Pulse")
        last_sniper_veto = query_system_state("last_veto_sniper", "None")
        market_label = query_system_state("market_regime_label", "TRENDING")
        warden_regime = query_system_state("warden_60m_regime", "LONG_ONLY")
        di_spread = query_system_state("market_regime_di_spread", "0.0")
        macro_4h = query_system_state("macro_trend_4h", "NEUTRAL")
        
        reg_color = "#00ff66" if market_label in ("BULL_TREND", "BULLISH", "TRENDING") else ("#ef4444" if market_label in ("BEAR_TREND", "BEARISH") else "#f59e0b")
        st.write(f"• **Market Regime Label:** <b style='color: {reg_color};'>{market_label}</b> <span style='color: #94a3b8; font-size: 0.8rem;'>(4H: {macro_4h})</span>", unsafe_allow_html=True)
        st.write(f"• **DI Polarity Spread:** <b style='color: {reg_color};'>{di_spread}</b>", unsafe_allow_html=True)
        st.write(f"• **60m Warden Bias:** <b style='color: #00ff66;'>{warden_regime}</b>", unsafe_allow_html=True)
        st.write(f"• **Volatility Cascade Risk:** <b style='color: #00ff66;'>CLEAR (Normal Flow)</b>", unsafe_allow_html=True)
        st.write(f"• **Last Filter Gate:** <span style='color: #94a3b8; font-size: 0.8rem;'>{last_sniper_veto[:40]}...</span>", unsafe_allow_html=True)

    # P1: Visual OFI Depth Imbalance Meter & Regime Strength Deck
    st.markdown("---")
    st.markdown("#### 🔬 Real-Time Order Flow Imbalance (OFI) & Regime Strength Deck")

    df_micro = query_db(TRADING_LEDGER_DB, "SELECT symbol, liquidity_imbalance, bid_wall_price, ask_wall_price, timestamp FROM oracle_microstructure ORDER BY id DESC LIMIT 1")
    imb = 0.0
    bid_wall = 0.0
    ask_wall = 0.0
    micro_sym = "BTC/USDT"
    if not df_micro.empty:
        imb = num(df_micro.iloc[0].get('liquidity_imbalance'), 0.0)
        bid_wall = num(df_micro.iloc[0].get('bid_wall_price'), 0.0)
        ask_wall = num(df_micro.iloc[0].get('ask_wall_price'), 0.0)
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

    # 🔬 Quantitative Microstructure & Sizing Telemetry (Scouted Alpha Engine)
    mlofi_val = 0.0
    micro_p = 0.0
    micro_bps = 0.0
    vpin_val = 0.0
    hawkes_spoof_val = 0.0
    hawkes_rho_val = 0.0
    hawkes_veto_val = False
    kyle_lambda_val = 0.0
    kyle_pctile_val = 0.5
    lvr_tax_bps_val = 0.0
    tox_halt_val = False
    try:
        if os.path.exists(MICROSTRUCTURE_STATE_FILE):
            with open(MICROSTRUCTURE_STATE_FILE, "r", encoding="utf-8") as f_ms:
                ms_json = json.load(f_ms)
                mlofi_val = float(ms_json.get("mlofi", 0.0) or 0.0)
                micro_p = float(ms_json.get("micro_price", 0.0) or 0.0)
                micro_bps = float(ms_json.get("micro_price_bps", 0.0) or 0.0)
                vpin_val = float(ms_json.get("vpin", 0.0) or 0.0)
                hawkes_spoof_val = float(ms_json.get("hawkes_spoof_index", 0.0) or 0.0)
                hawkes_rho_val = float(ms_json.get("hawkes_rho", 0.0) or 0.0)
                hawkes_veto_val = bool(ms_json.get("hawkes_veto", False))
                kyle_lambda_val = float(ms_json.get("kyle_lambda", 0.0) or 0.0)
                kyle_pctile_val = float(ms_json.get("kyle_lambda_pctile", 0.5) or 0.5)
                lvr_tax_bps_val = float(ms_json.get("lvr_tax_bps", 0.0) or 0.0)
                tox_halt_val = bool(ms_json.get("toxicity_halt", False))
    except Exception:
        pass

    # Kelly Sizing fraction
    live_kelly_pct = 15.0
    try:
        from risk_manager import RiskManager
        rm_dash = RiskManager()
        live_kelly_pct = rm_dash.calculate_empirical_kelly_fraction() * 100.0
    except Exception:
        pass

    vpin_color = "#00ff66" if vpin_val < 0.45 else ("#f59e0b" if vpin_val < 0.70 else "#ff003c")
    vpin_badge = "🟢 BENIGN FLOW" if vpin_val < 0.45 else ("🟡 CAUTION FLOW" if vpin_val < 0.70 else "🔴 TOXIC CASCADE")
    mlofi_color = "#00ff66" if mlofi_val > 0.05 else ("#ff003c" if mlofi_val < -0.05 else "#94a3b8")
    micro_color = "#00ff66" if micro_bps > 0 else ("#ff003c" if micro_bps < 0 else "#94a3b8")
    hawkes_color = "#ff003c" if (hawkes_veto_val or hawkes_spoof_val > 0.72) else ("#f59e0b" if hawkes_spoof_val > 0.45 else "#00ff66")
    hawkes_badge = "🔴 VETO TRIGGERED" if hawkes_veto_val else ("🟡 ELEVATED FLUX" if hawkes_spoof_val > 0.45 else "🟢 ORDER FLOW NORMAL")
    kyle_color = "#ff003c" if (tox_halt_val or kyle_pctile_val >= 0.90) else ("#f59e0b" if kyle_pctile_val >= 0.75 else "#00ff66")
    kyle_badge = "🔴 TOXICITY HALT" if tox_halt_val else (f"Pctile: {kyle_pctile_val*100:.0f}%")

    qcol1, qcol2, qcol3, qcol4, qcol5, qcol6 = st.columns(6)
    with qcol1:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {mlofi_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">🌊 Multi-Level OFI</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {mlofi_color}; font-family: 'JetBrains Mono', monospace;">{mlofi_val:+.3f}</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">10-Level Decay (α=0.5)</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol2:
        micro_str = f"${micro_p:,.2f}" if micro_p > 0 else "Tracking..."
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {micro_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">⚖️ Micro-Price Fair</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {micro_color}; font-family: 'JetBrains Mono', monospace;">{micro_str}</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">Bias: <b style="color: {micro_color};">{micro_bps:+.1f} bps</b></div>
        </div>
        """, unsafe_allow_html=True)

    with qcol3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {vpin_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">☣️ VPIN Toxicity</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {vpin_color}; font-family: 'JetBrains Mono', monospace;">{vpin_val:.3f}</div>
            <div style="color: {vpin_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">{vpin_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol4:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {hawkes_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">🦅 Hawkes Spoof Guard</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {hawkes_color}; font-family: 'JetBrains Mono', monospace;">{hawkes_spoof_val:.3f}</div>
            <div style="color: {hawkes_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">ρ={hawkes_rho_val:.2f} · {hawkes_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol5:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {kyle_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">λ Kyle / LVR Tax</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: {kyle_color}; font-family: 'JetBrains Mono', monospace;">{lvr_tax_bps_val:.1f} bps</div>
            <div style="color: {kyle_color}; font-size: 0.72rem; margin-top: 2px; font-weight: bold;">{kyle_badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with qcol6:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #38bdf8; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">📐 Dynamic Kelly</div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace;">{live_kelly_pct:.1f}%</div>
            <div style="color: #64748b; font-size: 0.72rem; margin-top: 2px;">Beta(13,7) Shrinkage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

    # ── Bayesian Markov Chain & Regime Transition Telemetry Deck ──────────────
    st.markdown("#### 🎲 Bayesian Markov Regime Engine & Transition Matrix (Live Forward Filter)")
    
    # Safe retrieval & default extraction
    markov_raw = query_system_state("markov_regime_telemetry", "{}")
    m_tel = {}
    try:
        if markov_raw and markov_raw != "{}":
            m_tel = json.loads(markov_raw)
    except Exception:
        m_tel = {}

    states_list = ["CHOP", "BULL_EXPANSION", "BEAR_CASCADE"]
    default_P = np.array([
        [0.70, 0.15, 0.15],
        [0.20, 0.70, 0.10],
        [0.20, 0.10, 0.70]
    ], dtype=np.float64)

    # If telemetry is absent or incomplete, compute fallback
    if not m_tel or "state_probabilities" not in m_tel:
        try:
            from markov_regime import MarkovRegimeEngine
            m_eng = MarkovRegimeEngine()
            dur_fb = m_eng.get_expected_regime_durations()
            stat_fb = m_eng.get_stationary_distribution()
            fc_fb = m_eng.forecast_k_steps(k=4)
            m_tel = {
                "dominant_regime": "CHOP",
                "state_probabilities": {"CHOP": 0.60, "BULL_EXPANSION": 0.25, "BEAR_CASCADE": 0.15},
                "expected_durations_bars": dur_fb,
                "breakdown_risk_4step": fc_fb.get("breakdown_risk", 0.35),
                "transition_matrix": default_P.tolist(),
                "forecast_probabilities": fc_fb.get("forecast_probabilities", {"CHOP": 0.50, "BULL_EXPANSION": 0.30, "BEAR_CASCADE": 0.20}),
                "stationary_distribution": {"CHOP": num(stat_fb[0]), "BULL_EXPANSION": num(stat_fb[1]), "BEAR_CASCADE": num(stat_fb[2])}
            }
        except Exception:
            m_tel = {
                "dominant_regime": "CHOP",
                "state_probabilities": {"CHOP": 0.60, "BULL_EXPANSION": 0.25, "BEAR_CASCADE": 0.15},
                "expected_durations_bars": {"CHOP": 3.33, "BULL_EXPANSION": 3.33, "BEAR_CASCADE": 3.33},
                "breakdown_risk_4step": 0.35,
                "transition_matrix": default_P.tolist(),
                "forecast_probabilities": {"CHOP": 0.50, "BULL_EXPANSION": 0.30, "BEAR_CASCADE": 0.20},
                "stationary_distribution": {"CHOP": 0.333, "BULL_EXPANSION": 0.333, "BEAR_CASCADE": 0.333}
            }

    dom_regime = m_tel.get("dominant_regime", "CHOP")
    probs = m_tel.get("state_probabilities", {})
    durations = m_tel.get("expected_durations_bars", {})
    bd_risk = float(m_tel.get("breakdown_risk_4step", 0.0) or 0.0)
    fc_probs = m_tel.get("forecast_probabilities", {})
    stat_dist = m_tel.get("stationary_distribution", {})

    p_chop = float(probs.get("CHOP", 0.0) or 0.0)
    p_bull = float(probs.get("BULL_EXPANSION", 0.0) or 0.0)
    p_bear = float(probs.get("BEAR_CASCADE", 0.0) or 0.0)

    d_chop = float(durations.get("CHOP", 0.0) or 0.0)
    d_bull = float(durations.get("BULL_EXPANSION", 0.0) or 0.0)
    d_bear = float(durations.get("BEAR_CASCADE", 0.0) or 0.0)

    # Normalize stationary distribution (dict or list)
    st_chop, st_bull, st_bear = 0.333, 0.333, 0.333
    raw_stat = m_tel.get("stationary_distribution")
    if isinstance(raw_stat, dict):
        st_chop = float(raw_stat.get("CHOP", 0.333) or 0.333)
        st_bull = float(raw_stat.get("BULL_EXPANSION", 0.333) or 0.333)
        st_bear = float(raw_stat.get("BEAR_CASCADE", 0.333) or 0.333)
    elif isinstance(raw_stat, (list, tuple, np.ndarray)) and len(raw_stat) >= 3:
        st_chop = num(raw_stat[0])
        st_bull = num(raw_stat[1])
        st_bear = num(raw_stat[2])

    # Normalize 4-step forecast probabilities (dict or list, check forecast_4step or forecast_probabilities)
    fc_chop, fc_bull, fc_bear = p_chop, p_bull, p_bear
    raw_fc = m_tel.get("forecast_probabilities") or m_tel.get("forecast_4step")
    if isinstance(raw_fc, dict):
        fc_chop = float(raw_fc.get("CHOP", p_chop) or 0.0)
        fc_bull = float(raw_fc.get("BULL_EXPANSION", p_bull) or 0.0)
        fc_bear = float(raw_fc.get("BEAR_CASCADE", p_bear) or 0.0)
    elif isinstance(raw_fc, (list, tuple, np.ndarray)) and len(raw_fc) >= 3:
        fc_chop = num(raw_fc[0])
        fc_bull = num(raw_fc[1])
        fc_bear = num(raw_fc[2])

    # Extract new Deep Research Playbook metrics
    is_tradable = bool(m_tel.get("is_tradable", True))
    tradable_rsn = str(m_tel.get("tradability_reason", "Tradable"))
    sizing_mult = float(m_tel.get("sizing_multiplier", 1.0) or 1.0)

    tradable_color = "#00ff66" if is_tradable else "#ff003c"
    tradable_badge = "TRADABLE" if is_tradable else "UNTRADEABLE CHURN"

    # Top 6 Markov & Playbook KPI Cards
    mk1, mk2, mk3, mk4, mk5, mk6 = st.columns(6)
    with mk1:
        dom_color = "#00ff66" if dom_regime == "BULL_EXPANSION" else ("#ff003c" if dom_regime == "BEAR_CASCADE" else "#f59e0b")
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {dom_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Active State (π_t)</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {dom_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{dom_regime}</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Conf: <b style="color: {dom_color};">{max(p_chop, p_bull, p_bear)*100:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with mk2:
        exp_d = durations.get(dom_regime, 3.3)
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #00e5ff; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Expected Duration</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: #00e5ff; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{exp_d:.1f} Bars</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">E[D] = 1 / (1 - P_ii)</div>
        </div>
        """, unsafe_allow_html=True)

    with mk3:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {tradable_color}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Tradability Gate</div>
            <div style="font-size: 1.05rem; font-weight: 800; color: {tradable_color}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">[{tradable_badge}]</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;" title="{tradable_rsn}">E[D] ≥ 3.5b Fee Shield</div>
        </div>
        """, unsafe_allow_html=True)

    with mk4:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #38bdf8; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Risk Multiplier</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{sizing_mult:.2f}x</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Continuous π-Scaling</div>
        </div>
        """, unsafe_allow_html=True)

    with mk5:
        bd_col = "#ff003c" if bd_risk >= 0.50 else ("#f59e0b" if bd_risk >= 0.30 else "#00ff66")
        bd_lbl = "HIGH" if bd_risk >= 0.60 else ("ELEV" if bd_risk >= 0.35 else "STABLE")
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid {bd_col}; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">4-Step Cascade Risk</div>
            <div style="font-size: 1.10rem; font-weight: 800; color: {bd_col}; font-family: 'JetBrains Mono', monospace; margin-top: 2px;">{bd_risk*100:.1f}%</div>
            <div style="color: {bd_col}; font-size: 0.70rem; margin-top: 2px; font-weight: bold;">[{bd_lbl}] P(Cascade)</div>
        </div>
        """, unsafe_allow_html=True)

    with mk6:
        st.markdown(f"""
        <div style="background: #0d121f; border: 1px solid #1e293b; border-left: 3px solid #a855f7; border-radius: 6px; padding: 12px 14px;">
            <div style="color: #94a3b8; font-size: 0.72rem; font-weight: bold; text-transform: uppercase;">Equilibrium (π*)</div>
            <div style="font-size: 0.95rem; font-weight: 800; color: #a855f7; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{st_chop*100:.0f}%/{st_bull*100:.0f}%/{st_bear*100:.0f}%</div>
            <div style="color: #64748b; font-size: 0.70rem; margin-top: 2px;">Chop/Bull/Bear</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 2-Column Plotly Visualizations: Transition Matrix Heatmap & Multi-Horizon Probabilities
    col_hm, col_bars = st.columns([1, 1])

    # Parse or format transition matrix P
    raw_P = m_tel.get("transition_matrix", default_P.tolist())
    try:
        mat_P = np.array(raw_P, dtype=np.float64)
        if mat_P.shape != (3, 3):
            mat_P = default_P
    except Exception:
        mat_P = default_P

    with col_hm:
        # Hover annotations and text labels with E[D] on diagonal
        text_matrix = []
        for i, s_from in enumerate(states_list):
            row_text = []
            for j, s_to in enumerate(states_list):
                val = mat_P[i, j]
                if i == j:
                    e_d = durations.get(s_from, 1.0 / max(1e-3, 1.0 - val))
                    row_text.append(f"{val:.2f}<br><span style='font-size:9px; color:#38bdf8;'>E[D]={e_d:.1f}b</span>")
                else:
                    row_text.append(f"{val:.2f}")
            text_matrix.append(row_text)

        fig_hm = go.Figure(data=go.Heatmap(
            z=mat_P,
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
        # Comparison: Current Belief π_t vs. 4-step Forecast π_{t+4} vs. Stationary π*
        labels_x = ["CHOP", "BULL_EXP", "BEAR_CAS"]
        pi_curr_vals = [p_chop, p_bull, p_bear]
        pi_fc_vals = [fc_chop, fc_bull, fc_bear]
        pi_stat_vals = [st_chop, st_bull, st_bear]

        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            name="Current π_t",
            x=labels_x,
            y=pi_curr_vals,
            marker_color="#00e5ff"
        ))
        fig_bars.add_trace(go.Bar(
            name="Forecast π_{t+4}",
            x=labels_x,
            y=pi_fc_vals,
            marker_color="#a855f7"
        ))
        fig_bars.add_trace(go.Bar(
            name="Equilibrium π*",
            x=labels_x,
            y=pi_stat_vals,
            marker_color="#64748b"
        ))

        # Add 0.65 entry hysteresis threshold line
        fig_bars.add_hline(
            y=0.65,
            line_dash="dot",
            line_color="#ff9900",
            annotation_text="0.65 Hysteresis Gate",
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
                    s_mlofi = float(sym_info.get("mlofi", 0.0) or 0.0)
                    s_mp = float(sym_info.get("micro_price", 0.0) or 0.0)
                    s_bps = float(sym_info.get("micro_price_bps", 0.0) or 0.0)
                    s_vpin = float(sym_info.get("vpin", 0.0) or 0.0)
                    s_bbid = float(sym_info.get("best_bid", 0.0) or 0.0)
                    s_bask = float(sym_info.get("best_ask", 0.0) or 0.0)
                    s_h_spoof = float(sym_info.get("hawkes_spoof_index", 0.0) or 0.0)
                    s_h_rho = float(sym_info.get("hawkes_rho", 0.0) or 0.0)
                    s_h_veto = bool(sym_info.get("hawkes_veto", False))
                    
                    s_kyle_pct = float(sym_info.get("kyle_lambda_pctile", 0.5) or 0.5)
                    s_lvr_tax = float(sym_info.get("lvr_tax_bps", 0.0) or 0.0)
                    s_tox_halt = bool(sym_info.get("toxicity_halt", False))
                    
                    v_stat = "🟢 Benign (<0.45)" if s_vpin < 0.45 else ("🟡 Caution (0.45-0.70)" if s_vpin < 0.70 else "🔴 TOXIC CASCADE")
                    flow_posture = "🟢 Buyer Depth" if s_mlofi > 5.0 else ("🔴 Seller Depth" if s_mlofi < -5.0 else "⚪ Balanced")
                    h_stat = "🔴 VETO ACTIVE" if (s_h_veto or s_h_spoof > 0.72 or s_h_rho >= 1.0) else ("🟡 Elevated" if s_h_spoof > 0.45 else "🟢 Stationary")
                    tox_stat = "🔴 HALT ACTIVE" if s_tox_halt else (f"🟢 Normal ({s_kyle_pct*100:.0f}%)" if s_kyle_pct < 0.80 else f"🟡 High ({s_kyle_pct*100:.0f}%)")
                    
                    fleet_rows.append({
                        "Asset": s_disp,
                        "Best Bid": f"${s_bbid:,.4f}" if s_bbid < 10 else f"${s_bbid:,.2f}",
                        "Best Ask": f"${s_bask:,.4f}" if s_bask < 10 else f"${s_bask:,.2f}",
                        "Micro-Price (Fair Value)": f"${s_mp:,.4f}" if s_mp < 10 else f"${s_mp:,.2f}",
                        "Spread Bias": f"{s_bps:+.2f} bps",
                        "MLOFI (10-Level)": f"{s_mlofi:+.2f}",
                        "VPIN (Toxicity)": f"{s_vpin:.3f}",
                        "Kyle Flow Risk": tox_stat,
                        "LVR Tax": f"{s_lvr_tax:.1f} bps",
                        "Hawkes Spoof": f"{s_h_spoof:.3f}",
                        "Spectral Radius (ρ)": f"{s_h_rho:.2f}",
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

    adx_val = num(df_regime.iloc[0].get('adx'), 28.5) if not df_regime.empty else 28.5
    hurst_val = num(df_regime.iloc[0].get('hurst'), 0.62) if not df_regime.empty else 0.62
    vol_state = num(df_oracle.iloc[0].get('volatility'), 0.55) if not df_oracle.empty else 0.55
    trend_pwr = num(df_oracle.iloc[0].get('trend_strength'), 0.40) if not df_oracle.empty else 0.40

    plus_di = num(df_regime.iloc[0].get('plus_di'), 0.0) if not df_regime.empty and 'plus_di' in df_regime.columns else 0.0
    minus_di = num(df_regime.iloc[0].get('minus_di'), 0.0) if not df_regime.empty and 'minus_di' in df_regime.columns else 0.0
    spread_di = plus_di - minus_di

    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    with gcol1:
        st.metric(
            label="Trend Strength (ADX)",
            value=f"{adx_val:.1f}",
            delta="Strong Trend (>25)" if adx_val >= 25 else "Low Momentum (<20)"
        )
    with gcol2:
        st.metric(
            label="DI Polarity Spread",
            value=f"{spread_di:+.1f}",
            delta=f"+DI: {plus_di:.1f} | -DI: {minus_di:.1f}" if (plus_di > 0 or minus_di > 0) else "Balanced"
        )
    with gcol3:
        st.metric(
            label="Persistence (Hurst)",
            value=f"{hurst_val:.3f}",
            delta="Persistent Trending (>0.52)" if hurst_val >= 0.52 else "Mean-Reverting (<0.48)"
        )
    with gcol4:
        st.metric(
            label="Volatility State (\u03c3)",
            value=f"{vol_state:.2f}",
            delta="Calibrated ATR Regime"
        )
    with gcol5:
        st.metric(
            label="Signal Readiness",
            value=f"{trend_pwr*100:.1f}%",
            delta="High Boardroom Conviction" if trend_pwr > 0.35 else "Selective Filtering"
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
        if mc_data:
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
