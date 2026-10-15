import sqlite3
import time
import signal
import threading
import logging
import json
import asyncio
import os
import math
from datetime import datetime, UTC, timezone
import yfinance as yf
import ai_engine
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MACRO_SENTINEL: %(message)s")
log = logging.getLogger("macro_sentinel")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_db_path(name: str) -> str:
    return os.path.join(BASE_DIR, name)

# P2-PATCH: Graceful shutdown event.
_stop_event = threading.Event()

def _handle_shutdown(signum, frame):
    log.info(f"[SHUTDOWN] Signal {signum} received. Completing current cycle then exiting...")
    _stop_event.set()

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)

def _parse_bool(val, default: bool = False) -> bool:
    """Defensively parses string or boolean values to avoid 'false' -> True bug."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "t"):
        return True
    if s in ("false", "0", "no", "f", ""):
        return False
    return default


KILLSWITCH_KEY = "macro_global_killswitch"
# A protective state may be inherited from the previous cycle; these are the only fields the
# macro synthesis is allowed to hold, and they are re-measured on every contract-valid verdict.
SYNTH_PROTECTIVE_FIELDS = ("market_posture", "macro_bias", "derivatives_skew_halt",
                           "derivatives_skew_regime", "derivatives_notes", "executive_summary")
ANCHOR_PROTECTIVE_FIELDS = ("btc_anchor_state", "alt_long_pause")
SYNTH_DEFENSIVE_DEFAULTS = {
    "market_posture": "TACTICAL_DEFENSE",
    "macro_bias": "NEUTRAL_CHOP",
    "derivatives_skew_halt": False,
    "derivatives_skew_regime": "BALANCED_CHOP",
    "derivatives_notes": "No contract-valid AI synthesis available; defensive baseline enforced.",
    "executive_summary": "AI synthesis unavailable; defensive baseline enforced.",
}
REGIME_PRECEDENCE_KEYS = ("market_regime_label", "super_bot_regime", "warden_60m_regime")


def _row_age_sec(raw_ts, now_epoch: float | None = None):
    """Age of a system_state row in seconds. None when the timestamp is missing or unusable."""
    if raw_ts is None or str(raw_ts).strip() == "":
        return None
    try:
        cleaned = str(raw_ts).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if now_epoch is None:
        now_epoch = time.time()
    return max(0.0, now_epoch - parsed.timestamp())


def _state_max_age_sec() -> float:
    try:
        value = float(getattr(config, "REGIME_MAX_AGE_SEC", 900.0))
    except (TypeError, ValueError):
        return 900.0
    if not math.isfinite(value) or value <= 0:
        return 900.0
    return value


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value or default)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def select_fresh_regime(rows: dict, now_epoch: float | None = None, max_age_sec: float | None = None):
    """Freshness-ranked regime precedence.

    ``rows`` maps a system_state key to the (value, updated_at) pair read from the ledger.
    The most recently published label wins. Absent, blank, unparseable or stale candidates are
    excluded outright, so a days-old warden label can never masquerade as the live regime.
    Returns (regime, winning_key, age_sec); ("UNKNOWN", None, None) when nothing is usable.
    """
    if max_age_sec is None:
        max_age_sec = _state_max_age_sec()
    best = None
    for key in REGIME_PRECEDENCE_KEYS:
        value, updated_at = rows.get(key, (None, None))
        if value is None or str(value).strip() == "":
            continue
        age = _row_age_sec(updated_at, now_epoch)
        if age is None:
            log.warning(f"[REGIME-PRECEDENCE] {key}: unusable updated_at {updated_at!r}; excluded.")
            continue
        if age > max_age_sec:
            log.warning(f"[REGIME-PRECEDENCE] {key}: stale by {age:.0f}s (max {max_age_sec:.0f}s); excluded.")
            continue
        if best is None or age < best[2]:
            best = (str(value).strip(), key, age)
    if best is None:
        return "UNKNOWN", None, None
    return best


def read_killswitch_latch(db_path: str):
    """The persisted killswitch latch as (flag, as_of). Read failure is treated as armed."""
    try:
        with sqlite3.connect(db_path, timeout=10.0) as conn:
            rows = dict(conn.execute(
                "SELECT key, value FROM system_state WHERE key IN (?, ?)",
                (KILLSWITCH_KEY, "macro_killswitch_asof")).fetchall())
    except Exception as e:
        log.warning(f"[MACRO-KILLSWITCH] Cannot read the persisted latch ({e}); assuming armed (fail-closed).")
        return 1, ""
    flag = 1 if _parse_bool(rows.get(KILLSWITCH_KEY), False) else 0
    return flag, str(rows.get("macro_killswitch_asof") or "")


def resolve_killswitch(data, stored_flag: int, stored_asof: str, now_iso: str) -> dict:
    """Asymmetric killswitch latch.

    Any cycle that measures SPY may ARM the killswitch, even when DXY is unmeasurable.
    Only a fully valid (data_ok) cycle may DISCHARGE it; every other outcome holds the last
    persisted value, so a Yahoo Finance outage cannot silently clear a market-wide halt.
    """
    measured = 1 if (data and _parse_bool(data.get("global_killswitch"), False)) else 0
    data_ok = bool(data and _parse_bool(data.get("data_ok"), False))
    if measured == 1:
        return {"active": 1, "state": "MEASURED" if data_ok else "MEASURED_PARTIAL", "asof": now_iso}
    if data_ok:
        return {"active": 0, "state": "MEASURED", "asof": now_iso}
    return {"active": 1 if stored_flag else 0, "state": "HELD", "asof": stored_asof or ""}


def normalize_macro_synth(ai_synth, held_synth, now_iso: str):
    """Hold the last protective synthesis when the AI could not answer.

    Returns (synth, was_held, asof). A provider failure must never soften the posture the fleet
    is already trading under, so the protective fields are inherited verbatim and marked stale.
    """
    if not isinstance(ai_synth, dict):
        # A non-dict provider answer carries no verdict at all: treat it as an outage.
        ai_synth = {}
        source = "fallback"
    else:
        # An unmarked payload is not a contract-valid verdict, so it may not soften anything.
        source = str(ai_synth.get("synthesis_source", "fallback")).strip().lower()
    if source == "ai":
        return ai_synth, False, now_iso

    if isinstance(held_synth, dict):
        for field in SYNTH_PROTECTIVE_FIELDS:
            ai_synth[field] = held_synth.get(field)
        asof = str(held_synth.get("_asof") or now_iso)
    else:
        for field, default_value in SYNTH_DEFENSIVE_DEFAULTS.items():
            ai_synth.setdefault(field, default_value)
        asof = ""
    ai_synth["posture_held"] = True
    return ai_synth, True, asof


def normalize_btc_anchor(anchor_res, held_anchor, now_iso: str):
    """Same asymmetric rule for the Bitcoin anchor: AI failure may not disarm ALT_LONG_PAUSE."""
    if not isinstance(anchor_res, dict):
        anchor_res = {}
        source = "fallback"
    else:
        source = str(anchor_res.get("anchor_source", "fallback")).strip().lower()
    if source == "ai":
        return anchor_res, False, now_iso
    if isinstance(held_anchor, dict):
        for field in ANCHOR_PROTECTIVE_FIELDS:
            anchor_res[field] = held_anchor.get(field)
        asof = str(held_anchor.get("_asof") or now_iso)
    else:
        anchor_res.setdefault("btc_anchor_state", "NEUTRAL_RANGE")
        anchor_res.setdefault("alt_long_pause", False)
        asof = ""
    anchor_res["anchor_held"] = True
    return anchor_res, True, asof


def apply_macro_overrides(ai_synth, active_event_playbook: str, killswitch_active: int, data_ok: bool):
    """Deterministic, tighten-only posture resolution. Returns (synth, is_skew_halt, skew_regime, skew_notes)."""
    # An event playbook may never LOOSEN posture over the deterministic gates. A bullish playbook
    # is therefore advisory: the expansion verdict has to come from the synthesis itself.
    if active_event_playbook == "SECULAR_SHORT_BREAKDOWN":
        ai_synth["macro_bias"] = "BEARISH"
        ai_synth["market_posture"] = "TACTICAL_DEFENSE"
    elif active_event_playbook == "BULL_MOMENTUM_EXPANSION":
        if killswitch_active == 1 or not data_ok:
            log.warning("[AI-MACRO] Refusing BULL_MOMENTUM_EXPANSION posture: Global Killswitch engaged or data invalid.")
            ai_synth["macro_bias"] = "NEUTRAL"
            ai_synth["market_posture"] = "TACTICAL_DEFENSE"
        else:
            log.info("[AI-MACRO] BULL_MOMENTUM_EXPANSION is advisory only; posture left to the synthesis (tighten-only).")

    is_skew_halt = _parse_bool(ai_synth.get("derivatives_skew_halt", False))
    skew_regime = str(ai_synth.get("derivatives_skew_regime", "BALANCED_CHOP") or "BALANCED_CHOP")
    skew_notes = str(ai_synth.get("derivatives_notes", "") or "")

    if killswitch_active == 1:
        ai_synth["market_posture"] = "RISK_OFF_KILLSWITCH"
        ai_synth["macro_bias"] = "DEFENSIVE_BEAR"
    elif is_skew_halt:
        ai_synth["market_posture"] = "TACTICAL_DEFENSE"
        ai_synth["macro_bias"] = "DEFENSIVE_CHOP"
    return ai_synth, is_skew_halt, skew_regime, skew_notes


def _persist_system_state(db_targets, values: dict, now_iso: str):
    """INSERT OR REPLACE a batch of system_state pairs into every ledger DB."""
    for db_target in db_targets:
        try:
            with sqlite3.connect(db_target, timeout=10.0) as conn:
                conn.execute("PRAGMA busy_timeout=10000;")
                with conn:
                    for key, value in values.items():
                        conn.execute(
                            "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, ?)",
                            (key, value, now_iso))
        except Exception as e:
            log.warning(f"Failed to persist macro state to {db_target}: {e}")



def init_db(db_path=None):
    if db_path is None:
        db_path = _get_db_path("trading_ledger.db")
    elif not os.path.isabs(db_path):
        db_path = _get_db_path(db_path)

    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_market_context (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    timestamp TEXT,
                    spy_price REAL,
                    uup_price REAL,
                    spy_1h_change REAL,
                    uup_1h_change REAL,
                    global_killswitch INTEGER
                )
            """)
            try:
                conn.execute("ALTER TABLE macro_market_context ADD COLUMN dxy_price REAL")
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute("ALTER TABLE macro_market_context ADD COLUMN macro_regime TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            # Append-only audit table for all macro dislocations and telemetry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_telemetry_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    spy_price REAL,
                    uup_price REAL,
                    dxy_price REAL,
                    spy_1h_change REAL,
                    uup_1h_change REAL,
                    avg_funding_rate_8h REAL,
                    global_killswitch INTEGER NOT NULL DEFAULT 0,
                    derivatives_skew_halt INTEGER NOT NULL DEFAULT 0,
                    alt_long_pause INTEGER NOT NULL DEFAULT 0,
                    macro_regime TEXT NOT NULL DEFAULT 'UNKNOWN',
                    market_posture TEXT NOT NULL DEFAULT 'TACTICAL_DEFENSE',
                    source TEXT DEFAULT 'macro_sentinel'
                )
            """)
        conn.close()
    except Exception as e:
        log.critical(f"Failed to init macro tables in {db_path}: {e}")
        raise

def _raw_fetch_macro():
    """Inner polling routine for yfinance SPY, UUP, and DXY with strict data validity gates."""
    try:
        spy = yf.Ticker("SPY").history(period="5d", interval="1h", timeout=8.0)
        uup = yf.Ticker("UUP").history(period="5d", interval="1h", timeout=8.0)
        
        dxy_latest = None
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d", timeout=8.0)
            if not dxy.empty:
                val = float(dxy['Close'].iloc[-1])
                if math.isfinite(val) and val > 0:
                    dxy_latest = val
            if dxy_latest is None:
                dxy = yf.Ticker("DXY").history(period="2d", timeout=8.0)
                if not dxy.empty:
                    val = float(dxy['Close'].iloc[-1])
                    if math.isfinite(val) and val > 0:
                        dxy_latest = val
        except Exception as e:
            log.warning(f"Failed to fetch DXY directly from Yahoo Finance: {e}.")
        # Note: Do not synthesize DXY via UUP * 3.48 multiplier because roll costs,
        # fees, and leverage drift cause false thresholds. DXY must be directly fetched.
                
        if len(spy) < 2 or len(uup) < 2:
            log.warning("Not enough hourly bars for SPY or UUP to compute 1h change.")
            return None
            
        spy_latest = float(spy['Close'].iloc[-1])
        spy_prev = float(spy['Close'].iloc[-2])
        if not math.isfinite(spy_latest) or not math.isfinite(spy_prev) or spy_prev <= 0:
            log.warning("Non-finite or invalid SPY price bars received.")
            return None
        spy_change = ((spy_latest - spy_prev) / spy_prev) * 100.0
        
        uup_latest = float(uup['Close'].iloc[-1])
        uup_prev = float(uup['Close'].iloc[-2])
        if not math.isfinite(uup_latest) or not math.isfinite(uup_prev) or uup_prev <= 0:
            log.warning("Non-finite or invalid UUP price bars received.")
            return None
        uup_change = ((uup_latest - uup_prev) / uup_prev) * 100.0
        
        # Guard: If DXY could not be resolved, do NOT fabricate 0.0.
        # DXY missing/zero makes dxy < 103.5 evaluate to True, which was falsely triggering RISK_ON_BULL.
        data_ok = True
        if dxy_latest is None or not math.isfinite(dxy_latest) or dxy_latest <= 50.0:
            log.warning(f"Invalid or missing DXY ({dxy_latest}). Marking data_ok=False and forcing regime UNKNOWN.")
            dxy_latest = None
            data_ok = False

        global_killswitch = 0
        if spy_change < -1.0:
            global_killswitch = 1
            log.critical(f"GLOBAL KILLSWITCH ACTIVATED: SPY dumped > 1.0%! (Change: {spy_change:.2f}%)")
        elif dxy_latest is not None and dxy_latest > 106.0:
            global_killswitch = 1
            log.critical(f"GLOBAL KILLSWITCH ACTIVATED: DXY dollar index has exceeded 106.0! (DXY: {dxy_latest:.2f})")

        # Macro Regime Classification:
        # If DXY or SPY is missing/invalid, regime must be UNKNOWN to avoid dangerous false bullish expansion.
        if global_killswitch:
            macro_regime = "RISK_OFF_KILLSWITCH"
        elif not data_ok or dxy_latest is None:
            macro_regime = "UNKNOWN"
        elif spy_change > 0.3 and dxy_latest < 103.5:
            macro_regime = "RISK_ON_BULL"
        elif spy_change < -0.4 or dxy_latest > 105.5:
            macro_regime = "DEFENSIVE_BEAR"
        else:
            macro_regime = "NEUTRAL"
        
        # Extended TradFi & Spot ETF Ingestion (HYG, LQD, IBIT)
        etf_macro_metrics = {}
        etf_gate_eval = {}
        try:
            hyg = yf.Ticker("HYG").history(period="15d", interval="1d", timeout=6.0)
            lqd = yf.Ticker("LQD").history(period="15d", interval="1d", timeout=6.0)
            ibit = yf.Ticker("IBIT").history(period="15d", interval="1d", timeout=6.0)
            
            if len(hyg) >= 5 and len(lqd) >= 5:
                from etf_macro_engine import MarketHoursAwareMacroGate
                gate = MarketHoursAwareMacroGate()
                # Compute stationary z-scores using daily history + 1h series
                spy_daily = yf.Ticker("SPY").history(period="60d", interval="1d", timeout=6.0)
                uup_daily = yf.Ticker("UUP").history(period="30d", interval="1d", timeout=6.0)
                
                if len(spy_daily) >= 20 and len(uup_daily) >= 5:
                    now_utc = datetime.now(timezone.utc)
                    gate.update_market_feed(spy_daily['Close'], uup_daily['Close'], hyg['Close'], lqd['Close'], now_utc)
                    etf_gate_eval = gate.evaluate_gate(now_utc)
                    etf_macro_metrics = etf_gate_eval.get("metrics", {})
        except Exception as etf_err:
            log.warning(f"[ETF-MACRO-NOTE] Optional ETF/Credit feed update: {etf_err}")

        return {
            'spy_price': spy_latest,
            'uup_price': uup_latest,
            'dxy_price': dxy_latest if dxy_latest is not None else 0.0,
            'dxy_valid': (dxy_latest is not None),
            'spy_1h_change': spy_change,
            'uup_1h_change': uup_change,
            'global_killswitch': global_killswitch,
            'macro_regime': macro_regime,
            'data_ok': data_ok,
            'etf_gate_eval': etf_gate_eval,
            'etf_macro_metrics': etf_macro_metrics,
        }
    except Exception as e:
        log.error(f"Error fetching macro data: {e}")
        return None

def fetch_macro_data():
    """
    Polls yfinance with non-blocking executor cleanup.
    Avoids ThreadPoolExecutor context manager wait-on-exit deadlock on TimeoutError.
    """
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_raw_fetch_macro)
    try:
        return future.result(timeout=10.0)
    except concurrent.futures.TimeoutError:
        log.warning("[MACRO-SENTINEL] Yahoo Finance API fetch timed out after 10.0s. Non-blocking cancellation triggered.")
        return None
    except Exception as e:
        log.error(f"[MACRO-SENTINEL] Unexpected error in macro worker: {e}")
        return None
    finally:
        # Crucial fix: wait=False, cancel_futures=True prevents hung threads from blocking python process
        executor.shutdown(wait=False, cancel_futures=True)

def fetch_derivatives_skew() -> dict:
    """
    Fetches live Bitget perpetual funding rates and open interest for BTC, ETH, and SOL.
    Uses fast REST queries with 4s timeout.
    """
    import urllib.request
    skew_data = {
        "symbols": {},
        "avg_funding_rate_8h": 0.0,
        "max_funding_rate_8h": 0.0,
        "valid_symbols_count": 0
    }
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    rates = []
    for s in symbols:
        fr_val = None
        oi_val = 0.0
        try:
            url_f = f"https://api.bitget.com/api/v2/mix/market/current-fund-rate?symbol={s}&productType=USDT-FUTURES"
            req = urllib.request.Request(url_f, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4.0) as res:
                d = json.loads(res.read().decode())
                if d.get("code") == "00000" and d.get("data"):
                    raw_fr = d["data"][0].get("fundingRate")
                    if raw_fr is not None:
                        val = float(raw_fr)
                        if math.isfinite(val):
                            fr_val = val
        except Exception as e:
            log.debug(f"Funding rate fetch note for {s}: {e}")

        try:
            url_oi = f"https://api.bitget.com/api/v2/mix/market/open-interest?symbol={s}&productType=USDT-FUTURES"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_oi, timeout=4.0) as res:
                d_oi = json.loads(res.read().decode())
                if d_oi.get("code") == "00000" and d_oi.get("data", {}).get("openInterestList"):
                    raw_oi = d_oi["data"]["openInterestList"][0].get("size")
                    if raw_oi is not None:
                        val_oi = float(raw_oi)
                        if math.isfinite(val_oi):
                            oi_val = val_oi
        except Exception as e:
            log.debug(f"Open interest fetch note for {s}: {e}")

        if fr_val is not None:
            rates.append(fr_val)
            skew_data["symbols"][s] = {
                "funding_rate_8h": fr_val,
                "funding_rate_8h_pct": f"{fr_val * 100.0:+.4f}%",
                "open_interest_contracts": oi_val
            }
        else:
            # Explicitly mark unverified rather than fabricating 0.0001
            skew_data["symbols"][s] = {
                "funding_rate_8h": 0.0,
                "funding_rate_8h_pct": "UNVERIFIED",
                "open_interest_contracts": oi_val
            }

    if rates:
        skew_data["avg_funding_rate_8h"] = sum(rates) / len(rates)
        skew_data["max_funding_rate_8h"] = max(rates)
        skew_data["valid_symbols_count"] = len(rates)

    return skew_data

def update_macro_context():
    # Initialize in both SQLite database files for full fleet visibility using absolute paths
    init_db(_get_db_path("trading_ledger.db"))
    init_db(_get_db_path("trading_data.db"))

    macro_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(macro_loop)
    iteration_count = 0

    held_synth = None
    held_anchor = None
    killswitch_log_state = None
    db_targets = [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]

    while not _stop_event.is_set():
        # Clean per-cycle state initialization to prevent cross-cycle variable leakage
        data = None
        data_ok = False
        deriv_skew = None
        ai_synth = None
        btc_price = 0.0
        warden_regime = "UNKNOWN"
        warden_regime_source = None
        btc_macro_trend = "UNKNOWN"
        active_event_playbook = "NORMAL_OPERATIONS"
        active_event_headline = ""
        active_event_horizon = "NEUTRAL"
        anchor_res = None
        hedge_rec = None
        ranking = None
        daily_bp = None
        now_iso = datetime.now(UTC).isoformat()

        log.info("Polling Yahoo Finance for Macro Context (SPY/UUP/DXY)...")
        data = fetch_macro_data()
        data_ok = bool(data and _parse_bool(data.get("data_ok"), False))
        stored_flag, stored_asof = read_killswitch_latch(_get_db_path("trading_data.db"))
        kill = resolve_killswitch(data, stored_flag, stored_asof, now_iso)
        killswitch_active = kill["active"]
        if killswitch_log_state != (killswitch_active, kill["state"]):
            killswitch_log_state = (killswitch_active, kill["state"])
            if killswitch_active == 1 and kill["state"] == "MEASURED_PARTIAL":
                log.critical("[MACRO-KILLSWITCH] ⚠️ ARMED on a partial macro cycle (DXY unmeasurable) and persisted "
                             "despite macro_data_ok=0. Discharge requires a fully valid cycle.")
            elif kill["state"] == "HELD":
                log.warning(f"[MACRO-KILLSWITCH] ⚠️ No authoritative measurement this cycle; holding "
                            f"killswitch={killswitch_active} from {kill['asof'] or 'never-persisted'}.")
            elif killswitch_active == 1:
                log.critical("[MACRO-KILLSWITCH] ⚠️ GLOBAL KILLSWITCH ARMED (measured this cycle).")
            else:
                log.info("[MACRO-KILLSWITCH] Killswitch clear on a fully valid macro cycle.")
        _persist_system_state(db_targets, {
            KILLSWITCH_KEY: str(killswitch_active),
            "macro_killswitch_state": kill["state"],
            "macro_killswitch_asof": kill["asof"],
        }, now_iso)

        if data_ok:
            for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                try:
                    conn = sqlite3.connect(db_target, timeout=30.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA busy_timeout=30000;")
                    with conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO macro_market_context (
                                id, timestamp, spy_price, uup_price, dxy_price,
                                spy_1h_change, uup_1h_change, global_killswitch, macro_regime
                            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            now_iso, data['spy_price'], data['uup_price'], data.get('dxy_price', 0.0),
                            data['spy_1h_change'], data['uup_1h_change'], data['global_killswitch'],
                            data['macro_regime']
                        ))
                        # Telemetry writes for state manager / HUD
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_global_killswitch', ?, ?)",
                                     (str(killswitch_active), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_regime', ?, ?)",
                                     (data['macro_regime'], now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_spy_1h_change', ?, ?)",
                                     (f"{data['spy_1h_change']:.2f}%", now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_dxy_price', ?, ?)",
                                     (f"{data['dxy_price']:.2f}", now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_data_ok', ?, ?)",
                                     ("1", now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_heartbeat', ?, ?)",
                                     (now_iso, now_iso))
                        # Persist stationary TradFi macro and ETF gate telemetry
                        etf_eval = data.get("etf_gate_eval", {})
                        if etf_eval:
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('etf_macro_telemetry', ?, ?)",
                                         (json.dumps(etf_eval), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('tradfi_macro_veto', ?, ?)",
                                         ("1" if etf_eval.get("veto_long_entries") else "0", now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('tradfi_confidence_weight', ?, ?)",
                                         (str(etf_eval.get("confidence_weight", 1.0)), now_iso))
                    conn.close()
                except Exception as e:
                    log.error(f"Failed to commit macro context to {db_target}: {e}")

            log_msg = f"SPY: ${data['spy_price']:.2f} ({data['spy_1h_change']:+.2f}%) | UUP: ${data['uup_price']:.2f} ({data['uup_1h_change']:+.2f}%) | DXY: {data.get('dxy_price', 0.0):.2f} | Regime: {data['macro_regime']}"
            if data['global_killswitch']:
                log.critical(f"GLOBAL KILLSWITCH STATUS: ACTIVE! {log_msg}")
            else:
                log.info(f"Macro Context Updated: {log_msg}")
        else:
            # SPY may still be measurable while DXY is not: _raw_fetch_macro already answers
            # UNKNOWN in that case, and RISK_OFF_KILLSWITCH when the dump trigger fired.
            fallback_regime = str(data.get('macro_regime') or 'UNKNOWN') if data else "UNKNOWN"
            log.warning(f"Macro data invalid or unavailable this cycle. Marking macro_data_ok=0, regime={fallback_regime}.")
            for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                try:
                    conn = sqlite3.connect(db_target, timeout=10.0)
                    with conn:
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_data_ok', ?, ?)",
                                     ("0", now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_regime', ?, ?)",
                                     (fallback_regime, now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_heartbeat', ?, ?)",
                                     (now_iso, now_iso))
                    conn.close()
                except Exception as de:
                    log.error(f"Failed to commit macro fallback status to {db_target}: {de}")

        # Economic Calendar Event Check
        try:
            from economic_calendar import EconomicCalendarGate
            econ_gate = EconomicCalendarGate(db_path=_get_db_path("trading_data.db"))
            econ_gate.update_system_state()
        except Exception as ee:
            log.warning(f"Economic calendar sync note in sentinel: {ee}")

        # Sovereign Macro & Derivatives Synthesis
        try:
            try:
                with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as c_data:
                    candidate_rows = {}
                    for rk in REGIME_PRECEDENCE_KEYS:
                        r_row = c_data.execute("SELECT value, updated_at FROM system_state WHERE key = ?", (rk,)).fetchone()
                        candidate_rows[rk] = (r_row[0], r_row[1]) if r_row else (None, None)
                    warden_regime, warden_regime_source, warden_age = select_fresh_regime(candidate_rows)
                    if warden_regime_source is None:
                        log.warning("[REGIME-PRECEDENCE] ⚠️ No fresh regime label among "
                                    f"{list(REGIME_PRECEDENCE_KEYS)}; feeding UNKNOWN to the synthesis.")
                    else:
                        log.debug(f"[REGIME-PRECEDENCE] Using {warden_regime_source} ({warden_age:.0f}s old): {warden_regime}")
                    r2 = c_data.execute("SELECT value, updated_at FROM system_state WHERE key = 'btc_macro_trend'").fetchone()
                    if r2 and r2[0] and _row_age_sec(r2[1]) is not None and _row_age_sec(r2[1]) <= _state_max_age_sec():
                        btc_macro_trend = str(r2[0])
                    r3 = c_data.execute("SELECT value FROM system_state WHERE key = 'btc_price'").fetchone()
                    if r3:
                        btc_price = _safe_float(r3[0], 0.0)
            except Exception as state_err:
                log.warning(f"[REGIME-PRECEDENCE] Could not read regime telemetry: {state_err}. Using UNKNOWN.")

            micro_path = os.path.join(BASE_DIR, "microstructure_state.json")
            if btc_price == 0.0 and os.path.exists(micro_path):
                try:
                    with open(micro_path, "r") as mf:
                        m_json = json.load(mf)
                        btc_price = float(m_json.get("best_bid", 0.0) or 0.0)
                except Exception:
                    pass

            try:
                with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as c_data:
                    ep_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_playbook'").fetchone()
                    if ep_row and ep_row[0]: active_event_playbook = ep_row[0]
                    eh_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_headline'").fetchone()
                    if eh_row and eh_row[0]: active_event_headline = eh_row[0]
                    ez_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_horizon'").fetchone()
                    if ez_row and ez_row[0]: active_event_horizon = ez_row[0]
            except Exception as ep_err:
                log.warning(f"[AI-MACRO] Event playbook telemetry unavailable, running NORMAL_OPERATIONS: {ep_err}")

            deriv_skew = fetch_derivatives_skew()
            # SPY/UUP are still measurable when only DXY failed, so they are passed through on a
            # partial cycle. DXY is sent as null instead of a fabricated 0.0 that would read as
            # "dollar collapsed" to the strategist.
            macro_packet = {
                "spy_price": data['spy_price'] if data else 0.0,
                "spy_1h_change": data['spy_1h_change'] if data else 0.0,
                "uup_price": data['uup_price'] if data else 0.0,
                "dxy_price": data.get('dxy_price') if (data and data.get('dxy_valid')) else None,
                "dxy_valid": bool(data.get('dxy_valid')) if data else False,
                "macro_regime": str(data.get('macro_regime') or "UNKNOWN") if data else "UNKNOWN",
                "global_killswitch": killswitch_active == 1,
                "killswitch_state": kill["state"],
                "data_ok": data_ok,
                "btc_price": btc_price,
                "warden_regime": warden_regime,
                "warden_regime_source": warden_regime_source or "STALE_OR_ABSENT",
                "btc_macro_trend": btc_macro_trend,
                "derivatives_skew": deriv_skew,
                "active_event_playbook": active_event_playbook,
                "active_event_horizon": active_event_horizon,
                "active_event_headline": active_event_headline
            }

            ai_synth = macro_loop.run_until_complete(ai_engine.generate_macro_synthesis(macro_packet, timeout_sec=15.0))
            # AI outage may not soften the posture the fleet is already trading under: inherit
            # the last protective synthesis and publish it as explicitly stale.
            ai_synth, synth_was_held, synth_asof = normalize_macro_synth(ai_synth, held_synth, now_iso)
            ai_synth, is_skew_halt, skew_regime, skew_notes = apply_macro_overrides(
                ai_synth, active_event_playbook, killswitch_active, data_ok)
            if not synth_was_held:
                held_synth = {field: ai_synth.get(field) for field in SYNTH_PROTECTIVE_FIELDS}
                held_synth["_asof"] = now_iso
            if synth_was_held:
                log.warning(f"[AI-MACRO] 🛑 Synthesis unavailable; holding posture "
                            f"{ai_synth.get('market_posture')}/{ai_synth.get('macro_bias')} from {synth_asof or 'defensive defaults'}.")

            for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                try:
                    with sqlite3.connect(db_target, timeout=10.0) as conn:
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_briefing', ?, ?)",
                                     (json.dumps(ai_synth), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_posture', ?, ?)",
                                     (ai_synth.get("market_posture", "TACTICAL_DEFENSE"), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_skew_posture', ?, ?)",
                                     (ai_synth.get("market_posture", "TACTICAL_DEFENSE"), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('derivatives_skew_regime', ?, ?)",
                                     (skew_regime, now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('derivatives_skew_halt', ?, ?)",
                                     (str(1 if is_skew_halt else 0), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('DERIVATIVES_SKEW_HALT', ?, ?)",
                                     (str(1 if is_skew_halt else 0), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('derivatives_skew_notes', ?, ?)",
                                     (skew_notes, now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('derivatives_funding_rates', ?, ?)",
                                     (json.dumps(deriv_skew.get("symbols", {}) if deriv_skew else {}), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_summary', ?, ?)",
                                     (ai_synth.get("executive_summary", ""), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_bias', ?, ?)",
                                     (ai_synth.get("macro_bias", "NEUTRAL_CHOP"), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_posture_held', ?, ?)",
                                     (str(1 if synth_was_held else 0), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_posture_asof', ?, ?)",
                                     (synth_asof or "", now_iso))
                except Exception as de:
                    log.warning(f"Error persisting macro briefing to {db_target}: {de}")

            if is_skew_halt:
                log.critical(f"[DERIVATIVES-SKEW-RADAR] ⚠️ DERIVATIVES SKEW HALT ACTIVATED! {skew_regime} | {skew_notes}")
            else:
                log.info(f"[DERIVATIVES-SKEW-RADAR] 📡 Skew Regime: {skew_regime} | Posture: {ai_synth.get('market_posture')} | Halt={is_skew_halt} | {skew_notes}")

            log.info(f"[AI-MACRO] Synthesis Complete: Posture={ai_synth.get('market_posture')} | Bias={ai_synth.get('macro_bias')} | {ai_synth.get('executive_summary', '')[:80]}")
        except Exception as syn_e:
            log.warning(f"[AI-MACRO] Periodic synthesis error: {syn_e}")

        # Bitcoin Anchor (Cross-Market Contagion Filter)
        try:
            btc_sym = "BTCUSDT"
            btc_fr = 0.0
            btc_oi = 0.0
            if deriv_skew and deriv_skew.get("symbols", {}).get(btc_sym):
                btc_fr = float(deriv_skew["symbols"][btc_sym].get("funding_rate_8h", 0.0) or 0.0)
                btc_oi = float(deriv_skew["symbols"][btc_sym].get("open_interest_contracts", 0.0) or 0.0)
                
            btc_ofi = 0.0
            micro_path = os.path.join(BASE_DIR, "microstructure_state.json")
            if os.path.exists(micro_path):
                try:
                    with open(micro_path, "r") as mf:
                        m_json = json.load(mf)
                        btc_ofi = float(m_json.get("symbols", {}).get("BTCUSDT", {}).get("normalized_ofi", 0.0) or 0.0)
                except Exception:
                    pass

            btc_anchor_packet = {
                "timestamp": now_iso,
                "btc_price": btc_price,
                "btc_macro_trend": btc_macro_trend,
                "warden_regime": warden_regime,
                "btc_funding_rate_8h": btc_fr,
                "btc_open_interest": btc_oi,
                "btc_normalized_ofi": btc_ofi,
                "spy_1h_change": data['spy_1h_change'] if data else 0.0,
                "dxy_price": data.get('dxy_price') if (data and data.get('dxy_valid')) else None
            }

            anchor_res = macro_loop.run_until_complete(ai_engine.evaluate_btc_anchor(btc_anchor_packet, timeout_sec=12.0))
            # An AI outage may not disarm the alt-coin long pause that is protecting the fleet.
            anchor_res, anchor_was_held, anchor_asof = normalize_btc_anchor(anchor_res, held_anchor, now_iso)
            if anchor_res:
                btc_state = anchor_res.get("btc_anchor_state", "HEALTHY_EXPANSION")
                alt_pause = _parse_bool(anchor_res.get("alt_long_pause", False))
                risk_score = _safe_float(anchor_res.get("contagion_risk_score"), 20.0)
                directive = anchor_res.get("anchor_directive", "Normal operations.")
                key_pivot = _safe_float(anchor_res.get("key_btc_pivot"), 0.0)
                if not anchor_was_held:
                    held_anchor = {field: anchor_res.get(field) for field in ANCHOR_PROTECTIVE_FIELDS}
                    held_anchor["_asof"] = now_iso
                else:
                    log.warning(f"[BTC-ANCHOR] 🛑 Anchor unavailable; holding {btc_state} / "
                                f"ALT_LONG_PAUSE={int(alt_pause)} from {anchor_asof or 'baseline'}.")

                log.info(f"[BTC-ANCHOR] ⚓ BTC State: {btc_state} | AltLongPause: {alt_pause} | ContagionRisk: {risk_score}/100 | {directive}")

                for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                    try:
                        with sqlite3.connect(db_target, timeout=10.0) as conn:
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_anchor_state', ?, ?)",
                                         (btc_state, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ALT_LONG_PAUSE', ?, ?)",
                                         (str(1 if alt_pause else 0), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('alt_long_pause', ?, ?)",
                                         (str(1 if alt_pause else 0), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_contagion_risk_score', ?, ?)",
                                         (str(risk_score), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_anchor_directive', ?, ?)",
                                         (directive, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_key_pivot', ?, ?)",
                                         (str(key_pivot), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_btc_anchor', ?, ?)",
                                         (json.dumps(anchor_res), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_anchor_held', ?, ?)",
                                         (str(1 if anchor_was_held else 0), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('btc_anchor_asof', ?, ?)",
                                         (anchor_asof or "", now_iso))
                    except Exception as de:
                        log.warning(f"Error persisting BTC anchor to {db_target}: {de}")
        except Exception as anchor_err:
            log.warning(f"[BTC-ANCHOR] Periodic Bitcoin anchor error: {anchor_err}")

        # AI Delta-Neutral Hedging Oracle
        try:
            open_pos_list = []
            net_long_usd = 0.0
            net_short_usd = 0.0
            try:
                with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as p_conn:
                    for tbl in ["open_positions", "berserker_open_positions"]:
                        try:
                            rows = p_conn.execute(f"SELECT symbol, side, qty, entry_price FROM {tbl}").fetchall()
                            for r in rows:
                                p_sym, p_side, p_qty, p_ep = r[0], str(r[1]).lower(), float(r[2] or 0), float(r[3] or 0)
                                notional = p_qty * p_ep
                                if p_side == "long":
                                    net_long_usd += notional
                                else:
                                    net_short_usd += notional
                                open_pos_list.append({
                                    "symbol": p_sym,
                                    "side": p_side,
                                    "notional_usd": round(notional, 2)
                                })
                        except Exception:
                            pass
            except Exception:
                pass

            net_delta_dollars = round(net_long_usd - net_short_usd, 2)
            posture = ai_synth.get("market_posture", "TACTICAL_DEFENSE") if ai_synth else "NEUTRAL"
            
            weakest_list = ["ETH/USDT", "XRP/USDT"]
            try:
                with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as a_conn:
                    r_av = a_conn.execute("SELECT value FROM system_state WHERE key = 'ai_asset_avoid'").fetchone()
                    if r_av and r_av[0]:
                        weakest_list = [x.strip() for x in r_av[0].split(",") if x.strip()]
            except Exception:
                pass

            hedging_context = {
                "timestamp": now_iso,
                "net_delta_dollars": net_delta_dollars,
                "gross_long_dollars": round(net_long_usd, 2),
                "gross_short_dollars": round(net_short_usd, 2),
                "open_positions": open_pos_list,
                "macro_regime": str(data.get('macro_regime') or 'UNKNOWN') if data else "UNKNOWN",
                "macro_posture": posture,
                "global_killswitch": killswitch_active == 1,
                "killswitch_state": kill["state"],
                "spy_1h_change": data.get('spy_1h_change', 0.0) if data else 0.0,
                "dxy_price": data.get('dxy_price') if (data and data.get('dxy_valid')) else None,
                "weakest_fleet_assets": weakest_list
            }

            hedge_rec = macro_loop.run_until_complete(ai_engine.calculate_delta_neutral_hedge(hedging_context, timeout_sec=15.0))
            if hedge_rec:
                log.info(f"[HEDGING-ORACLE] 🛡️ Shield={hedge_rec.get('risk_shield_level')} | HedgeReq={hedge_rec.get('hedge_required')} | Asset={hedge_rec.get('hedge_asset')} ({hedge_rec.get('hedge_side')}) | Target=${hedge_rec.get('target_short_notional')} | Rationale={hedge_rec.get('rationale')}")
                for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                    try:
                        with sqlite3.connect(db_target, timeout=10.0) as conn:
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_hedging_oracle', ?, ?)",
                                         (json.dumps(hedge_rec), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_hedging_shield_level', ?, ?)",
                                         (hedge_rec.get('risk_shield_level', 'GREEN_SHIELD'), now_iso))
                    except Exception as de:
                        log.warning(f"Error persisting hedging plan to {db_target}: {de}")
        except Exception as hedge_e:
            log.warning(f"[HEDGING-ORACLE] Periodic hedging oracle error: {hedge_e}")

        # Append-Only Telemetry Table Write: commit comprehensive row for auditability
        try:
            active_kill = killswitch_active
            skew_halt_flag = 1 if (ai_synth and _parse_bool(ai_synth.get("derivatives_skew_halt", False))) else 0
            alt_pause_flag = 1 if (anchor_res and _parse_bool(anchor_res.get("alt_long_pause", False))) else 0
            curr_regime = str(data['macro_regime']) if (data and data.get('macro_regime')) else "UNKNOWN"
            curr_posture = ai_synth.get('market_posture', 'TACTICAL_DEFENSE') if ai_synth else "TACTICAL_DEFENSE"
            avg_fr = deriv_skew.get("avg_funding_rate_8h", 0.0) if deriv_skew else 0.0

            for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                try:
                    conn = sqlite3.connect(db_target, timeout=10.0)
                    with conn:
                        conn.execute("""
                            INSERT INTO macro_telemetry_history (
                                timestamp, spy_price, uup_price, dxy_price, spy_1h_change, uup_1h_change,
                                avg_funding_rate_8h, global_killswitch, derivatives_skew_halt,
                                alt_long_pause, macro_regime, market_posture, source
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            now_iso,
                            data['spy_price'] if data else None,
                            data['uup_price'] if data else None,
                            data.get('dxy_price') if (data and data.get('dxy_valid')) else None,
                            data['spy_1h_change'] if data else None,
                            data['uup_1h_change'] if data else None,
                            avg_fr,
                            active_kill,
                            skew_halt_flag,
                            alt_pause_flag,
                            curr_regime,
                            curr_posture,
                            'macro_sentinel'
                        ))
                    conn.close()
                except Exception as he:
                    log.warning(f"Failed to append macro_telemetry_history to {db_target}: {he}")
        except Exception as tel_err:
            log.warning(f"Telemetry history commit error: {tel_err}")

        # Chief Strategist Asset Hierarchy Leaderboard (Every 2 hours = 8 iterations)
        if iteration_count % 8 == 0:
            try:
                log.info("[CHIEF-STRATEGIST] Polling Fleet Metrics for 2-Hour Institutional Asset Ranking...")
                fleet_symbols = getattr(config, "SYMBOLS", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT'])
                assets_metrics = {}
                for s in fleet_symbols[:10]:
                    clean_s = s.replace('/', '').replace(':USDT', '').replace('-', '').upper()
                    asset_regime = "TRENDING"
                    asset_adx = 25.0
                    asset_hurst = 0.52
                    try:
                        with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as c_m:
                            r_lbl = c_m.execute("SELECT value FROM system_state WHERE key = ?", (f"regime_{clean_s}",)).fetchone()
                            if r_lbl: asset_regime = r_lbl[0]
                            r_adx = c_m.execute("SELECT value FROM system_state WHERE key = ?", (f"adx_{clean_s}",)).fetchone()
                            if r_adx: asset_adx = float(r_adx[0] or 25.0)
                            r_h = c_m.execute("SELECT hurst FROM active_fleet_config WHERE symbol LIKE ?", (f"%{clean_s[:3]}%",)).fetchone()
                            if r_h and r_h[0]: asset_hurst = float(r_h[0] or 0.52)
                    except Exception:
                        pass
                        
                    assets_metrics[s] = {
                        "regime": asset_regime,
                        "adx": asset_adx,
                        "hurst": asset_hurst
                    }

                fleet_context = {
                    "timestamp": now_iso,
                    "btc_price": btc_price,
                    "macro_posture": ai_synth.get("market_posture", "TACTICAL_DEFENSE") if ai_synth else "NEUTRAL",
                    "assets": assets_metrics
                }

                ranking = macro_loop.run_until_complete(ai_engine.rank_asset_hierarchy(fleet_context, timeout_sec=20.0))
                if ranking:
                    rank_str = json.dumps(ranking)
                    leaders = ", ".join(ranking.get("priority_leaders", []))
                    avoid = ", ".join(ranking.get("avoid_chop_list", []))
                    log.info(f"[CHIEF-STRATEGIST] 👑 Asset Hierarchy Updated: Leaders=[{leaders}] | Avoid=[{avoid}] | Rationale={ranking.get('fleet_focus_rationale')}")

                    for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                        try:
                            with sqlite3.connect(db_target, timeout=10.0) as conn:
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_asset_rankings', ?, ?)",
                                             (rank_str, now_iso))
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_asset_leaders', ?, ?)",
                                             (leaders, now_iso))
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_asset_avoid', ?, ?)",
                                             (avoid, now_iso))
                        except Exception as de:
                            log.warning(f"Error persisting asset rankings to {db_target}: {de}")
            except Exception as rank_err:
                log.warning(f"[CHIEF-STRATEGIST] Periodic ranking error: {rank_err}")

        # AI 6-Hour Strategic Macro Blueprint & Key Level Roadmap
        try:
            now_utc = datetime.now(UTC)
            today_utc = now_utc.strftime("%Y-%m-%d")
            session_idx = now_utc.hour // 6
            session_start_h = session_idx * 6
            session_tag = f"{today_utc}_S{session_idx+1}_{session_start_h:02d}hUTC"
            session_names = [
                "Asia / Pacific Session (00:00 UTC)",
                "Europe / London Session (06:00 UTC)",
                "US Wall Street Open (12:00 UTC)",
                "US Afternoon / Asia Pre-Open (18:00 UTC)"
            ]
            session_name = session_names[session_idx]

            needs_blueprint = False
            try:
                with sqlite3.connect(_get_db_path("trading_data.db"), timeout=5.0) as b_conn:
                    row_bp = b_conn.execute("SELECT value, updated_at FROM system_state WHERE key = 'ai_daily_blueprint'").fetchone()
                    if not row_bp or not row_bp[0]:
                        needs_blueprint = True
                    else:
                        bp_data = json.loads(row_bp[0])
                        stored_tag = bp_data.get("session_tag")
                        last_upd = row_bp[1]
                        if stored_tag != session_tag:
                            needs_blueprint = True
                        elif last_upd:
                            try:
                                dt_clean = str(last_upd).replace("Z", "+00:00")
                                dt_last = datetime.fromisoformat(dt_clean)
                                if dt_last.tzinfo is None:
                                    dt_last = dt_last.replace(tzinfo=UTC)
                                if (now_utc - dt_last).total_seconds() >= 6 * 3600:
                                    needs_blueprint = True
                            except Exception:
                                pass
            except Exception:
                needs_blueprint = True

            if needs_blueprint:
                log.info(f"[MACRO-BLUEPRINT] Generating fresh 6-Hour AI Strategic Blueprint for session [{session_tag}: {session_name}]...")
                fleet_symbols = getattr(config, "SYMBOLS", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT'])
                
                live_prices = {}
                micro_path = os.path.join(BASE_DIR, "microstructure_state.json")
                if os.path.exists(micro_path):
                    try:
                        with open(micro_path, "r") as msf:
                            ms_data = json.load(msf)
                            sym_dict = ms_data.get("symbols", {})
                            for s in fleet_symbols[:10]:
                                clean = s.replace('/', '').replace(':USDT', '').replace('-', '')
                                if clean in sym_dict:
                                    live_prices[s] = sym_dict[clean].get("best_bid") or sym_dict[clean].get("best_ask")
                    except Exception:
                        pass

                bp_snapshot = {
                    "date": today_utc,
                    "session_tag": session_tag,
                    "session_name": session_name,
                    "btc_price": live_prices.get("BTC/USDT", btc_price),
                    "asset_current_prices": live_prices,
                    "macro_regime": str(data.get('macro_regime') or 'UNKNOWN') if data else "UNKNOWN",
                    "spy_price": data.get('spy_price', 0.0) if data else 0.0,
                    "dxy_price": data.get('dxy_price') if (data and data.get('dxy_valid')) else None,
                    "global_killswitch": killswitch_active == 1,
                    "fleet_symbols": fleet_symbols[:10]
                }
                daily_bp = macro_loop.run_until_complete(ai_engine.generate_daily_market_blueprint(fleet_symbols[:10], bp_snapshot, timeout_sec=25.0))
                if daily_bp:
                    daily_bp["session_tag"] = session_tag
                    daily_bp["session_name"] = session_name
                    bp_json = json.dumps(daily_bp)
                    log.info(f"[MACRO-BLUEPRINT] 🗺️ 6-Hour Blueprint Generated [{session_tag}] ({daily_bp.get('macro_bias')}): {daily_bp.get('daily_narrative', '')[:80]}...")
                    for db_target in [_get_db_path("trading_ledger.db"), _get_db_path("trading_data.db")]:
                        try:
                            with sqlite3.connect(db_target, timeout=10.0) as conn:
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_daily_blueprint', ?, ?)",
                                             (bp_json, now_iso))
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_daily_blueprint_bias', ?, ?)",
                                             (daily_bp.get('macro_bias', 'DEFENSIVE_CHOP'), now_iso))
                        except Exception as dbe:
                            log.warning(f"Error persisting 6-hour blueprint to {db_target}: {dbe}")
        except Exception as bp_err:
            log.warning(f"[MACRO-BLUEPRINT-NOTE] Error generating 6-hour blueprint: {bp_err}")

        iteration_count += 1
        
        log.info("Sleeping for 15 minutes (interruptible)...")
        _stop_event.wait(timeout=15 * 60)

    try:
        macro_loop.close()
    except Exception:
        pass
    log.info("[SHUTDOWN] Macro Sentinel exiting cleanly.")

if __name__ == "__main__":
    update_macro_context()
