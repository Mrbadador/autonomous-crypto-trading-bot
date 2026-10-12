import sqlite3
import time
import signal
import threading
import logging
import json
import asyncio
import os
from datetime import datetime, UTC
import yfinance as yf
import ai_engine
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] MACRO_SENTINEL: %(message)s")
log = logging.getLogger("macro_sentinel")

# P2-PATCH: Graceful shutdown event.
# Without this, SIGTERM from main.py killed the process mid-sleep, leaving the
# killswitch row in whatever state it was last written. Now the loop checks this
# flag and exits cleanly after the current DB write completes.
_stop_event = threading.Event()

def _handle_shutdown(signum, frame):
    log.info(f"[SHUTDOWN] Signal {signum} received. Completing current cycle then exiting...")
    _stop_event.set()

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


def init_db(db_path="trading_ledger.db"):
    try:
        conn = sqlite3.connect(db_path)
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
                pass # Already exists
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
        conn.close()
    except Exception as e:
        log.critical(f"Failed to init macro_market_context table in {db_path}: {e}")
        raise

def _raw_fetch_macro():
    """Inner polling routine for yfinance SPY, UUP, and DXY."""
    try:
        spy = yf.Ticker("SPY").history(period="5d", interval="1h")
        uup = yf.Ticker("UUP").history(period="5d", interval="1h")
        
        dxy_latest = 0.0
        try:
            dxy = yf.Ticker("DX-Y.NYB").history(period="2d")
            if not dxy.empty:
                dxy_latest = float(dxy['Close'].iloc[-1])
            else:
                dxy = yf.Ticker("DXY").history(period="2d")
                if not dxy.empty:
                    dxy_latest = float(dxy['Close'].iloc[-1])
        except Exception as e:
            log.warning(f"Failed to fetch DXY from Yahoo Finance: {e}. Fallback to UUP estimate.")
            
        if dxy_latest == 0.0 and len(uup) > 0:
            uup_latest = float(uup['Close'].iloc[-1])
            dxy_latest = uup_latest * 3.48
            
        if len(spy) < 2 or len(uup) < 2:
            log.warning("Not enough hourly bars to compute 1h change.")
            return None
            
        spy_latest = float(spy['Close'].iloc[-1])
        spy_prev = float(spy['Close'].iloc[-2])
        spy_change = ((spy_latest - spy_prev) / spy_prev) * 100.0
        
        uup_latest = float(uup['Close'].iloc[-1])
        uup_prev = float(uup['Close'].iloc[-2])
        uup_change = ((uup_latest - uup_prev) / uup_prev) * 100.0
        
        global_killswitch = 0
        if spy_change < -1.0:
            global_killswitch = 1
            log.critical(f"GLOBAL KILLSWITCH ACTIVATED: SPY dumped > 1.0%! (Change: {spy_change:.2f}%)")
        elif dxy_latest > 106.0:
            global_killswitch = 1
            log.critical(f"GLOBAL KILLSWITCH ACTIVATED: DXY dollar index has exceeded 106.0! (DXY: {dxy_latest:.2f})")

        # Macro Regime Classification
        if global_killswitch:
            macro_regime = "RISK_OFF_KILLSWITCH"
        elif spy_change > 0.3 and dxy_latest < 103.5:
            macro_regime = "RISK_ON_BULL"
        elif spy_change < -0.4 or dxy_latest > 105.5:
            macro_regime = "DEFENSIVE_BEAR"
        else:
            macro_regime = "NEUTRAL"
        
        return {
            'spy_price': spy_latest,
            'uup_price': uup_latest,
            'dxy_price': dxy_latest,
            'spy_1h_change': spy_change,
            'uup_1h_change': uup_change,
            'global_killswitch': global_killswitch,
            'macro_regime': macro_regime
        }
    except Exception as e:
        log.error(f"Error fetching macro data: {e}")
        return None

def fetch_macro_data():
    """Polls yfinance with a hard 10-second timeout to prevent thread hangs during Yahoo API degradation."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_raw_fetch_macro)
        try:
            return future.result(timeout=10.0)
        except concurrent.futures.TimeoutError:
            log.warning("[MACRO-SENTINEL] Yahoo Finance API fetch timed out after 10.0s. Preserving last known state.")
        except Exception as e:
            log.error(f"[MACRO-SENTINEL] Unexpected error in macro worker: {e}")
            return None

def fetch_derivatives_skew() -> dict:
    """
    Fetches live Bitget perpetual funding rates and open interest for BTC, ETH, and SOL.
    Uses fast REST queries with 4s timeout.
    """
    import urllib.request
    skew_data = {
        "symbols": {},
        "avg_funding_rate_8h": 0.0,
        "max_funding_rate_8h": 0.0
    }
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    rates = []
    for s in symbols:
        fr_val = 0.0001
        oi_val = 0.0
        try:
            url_f = f"https://api.bitget.com/api/v2/mix/market/current-fund-rate?symbol={s}&productType=USDT-FUTURES"
            req = urllib.request.Request(url_f, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=4.0) as res:
                d = json.loads(res.read().decode())
                if d.get("code") == "00000" and d.get("data"):
                    fr_val = float(d["data"][0].get("fundingRate", 0.0) or 0.0)
        except Exception as e:
            log.debug(f"Funding rate fetch note for {s}: {e}")

        try:
            url_oi = f"https://api.bitget.com/api/v2/mix/market/open-interest?symbol={s}&productType=USDT-FUTURES"
            req_oi = urllib.request.Request(url_oi, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_oi, timeout=4.0) as res:
                d_oi = json.loads(res.read().decode())
                if d_oi.get("code") == "00000" and d_oi.get("data", {}).get("openInterestList"):
                    oi_val = float(d_oi["data"]["openInterestList"][0].get("size", 0.0) or 0.0)
        except Exception as e:
            log.debug(f"Open interest fetch note for {s}: {e}")

        rates.append(fr_val)
        skew_data["symbols"][s] = {
            "funding_rate_8h": fr_val,
            "funding_rate_8h_pct": f"{fr_val * 100.0:+.4f}%",
            "open_interest_contracts": oi_val
        }

    if rates:
        skew_data["avg_funding_rate_8h"] = sum(rates) / len(rates)
        skew_data["max_funding_rate_8h"] = max(rates)

    return skew_data

def update_macro_context():
    # Initialize in both SQLite database files for full fleet visibility
    init_db("trading_ledger.db")
    init_db("trading_data.db")

    macro_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(macro_loop)
    iteration_count = 0

    while not _stop_event.is_set():
        log.info("Polling Yahoo Finance for Macro Context (SPY/UUP/DXY)...")
        data = fetch_macro_data()
        if data:
            timestamp = datetime.now(UTC).isoformat()
            for db_target in ["trading_ledger.db", "trading_data.db"]:
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
                            timestamp, data['spy_price'], data['uup_price'], data.get('dxy_price', 0.0),
                            data['spy_1h_change'], data['uup_1h_change'], data['global_killswitch'],
                            data['macro_regime']
                        ))
                        # Telemetry writes for state manager / HUD
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_global_killswitch', ?, ?)",
                                     (str(data['global_killswitch']), timestamp))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_regime', ?, ?)",
                                     (data['macro_regime'], timestamp))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_spy_1h_change', ?, ?)",
                                     (f"{data['spy_1h_change']:.2f}%", timestamp))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('macro_dxy_price', ?, ?)",
                                     (f"{data['dxy_price']:.2f}", timestamp))
                    conn.close()
                except Exception as e:
                    log.error(f"Failed to commit macro context to {db_target}: {e}")

            log_msg = f"SPY: ${data['spy_price']:.2f} ({data['spy_1h_change']:+.2f}%) | UUP: ${data['uup_price']:.2f} ({data['uup_1h_change']:+.2f}%) | DXY: {data.get('dxy_price', 0.0):.2f} | Regime: {data['macro_regime']}"
            if data['global_killswitch']:
                log.critical(f"GLOBAL KILLSWITCH STATUS: ACTIVE! {log_msg}")
            else:
                log.info(f"Macro Context Updated: {log_msg}")

        # 7.4 Economic Calendar Event Check
        try:
            from economic_calendar import EconomicCalendarGate
            econ_gate = EconomicCalendarGate(db_path="trading_data.db")
            econ_gate.update_system_state()
        except Exception as ee:
            log.warning(f"Economic calendar sync note in sentinel: {ee}")

        # V37.0 Autonomous AI Sovereign Macro & Derivatives Synthesis (Google Gemini Flash)
        try:
            # Gather crypto market context from system_state
            warden_regime = "UNKNOWN"
            btc_macro_trend = "UNKNOWN"
            btc_price = 0.0
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as c_data:
                    # Prioritize live regime classifier & super bot regime over obsolete warden keys
                    for rk in ['market_regime_label', 'super_bot_regime', 'warden_60m_regime']:
                        r_row = c_data.execute("SELECT value FROM system_state WHERE key = ?", (rk,)).fetchone()
                        if r_row and r_row[0]:
                            warden_regime = r_row[0]
                            break
                    r2 = c_data.execute("SELECT value FROM system_state WHERE key = 'btc_macro_trend'").fetchone()
                    if r2: btc_macro_trend = r2[0]
                    r3 = c_data.execute("SELECT value FROM system_state WHERE key = 'btc_price'").fetchone()
                    if r3 and r3[0]: btc_price = float(r3[0])
            except Exception:
                pass

            if btc_price == 0.0 and os.path.exists("microstructure_state.json"):
                try:
                    with open("microstructure_state.json", "r") as mf:
                        m_json = json.load(mf)
                        btc_price = float(m_json.get("best_bid", 0.0) or 0.0)
                except Exception:
                    pass

            # V40.0 Ingest active event catalyst & headline
            active_event_playbook = "NORMAL_OPERATIONS"
            active_event_headline = ""
            active_event_horizon = "NEUTRAL"
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as c_data:
                    ep_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_playbook'").fetchone()
                    if ep_row and ep_row[0]: active_event_playbook = ep_row[0]
                    eh_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_headline'").fetchone()
                    if eh_row and eh_row[0]: active_event_headline = eh_row[0]
                    ez_row = c_data.execute("SELECT value FROM system_state WHERE key = 'ai_event_horizon'").fetchone()
                    if ez_row and ez_row[0]: active_event_horizon = ez_row[0]
            except Exception:
                pass

            deriv_skew = fetch_derivatives_skew()
            macro_packet = {
                "spy_price": data['spy_price'] if data else 0.0,
                "spy_1h_change": data['spy_1h_change'] if data else 0.0,
                "uup_price": data['uup_price'] if data else 0.0,
                "dxy_price": data.get('dxy_price', 0.0) if data else 0.0,
                "macro_regime": data['macro_regime'] if data else "NEUTRAL",
                "global_killswitch": bool(data['global_killswitch']) if data else False,
                "btc_price": btc_price,
                "warden_regime": warden_regime,
                "btc_macro_trend": btc_macro_trend,
                "derivatives_skew": deriv_skew,
                "active_event_playbook": active_event_playbook,
                "active_event_horizon": active_event_horizon,
                "active_event_headline": active_event_headline
            }

            ai_synth = macro_loop.run_until_complete(ai_engine.generate_macro_synthesis(macro_packet, timeout_sec=15.0))
            if not isinstance(ai_synth, dict):
                ai_synth = {
                    "macro_bias": "NEUTRAL",
                    "market_posture": "TACTICAL_DEFENSE",
                    "derivatives_skew_halt": False,
                    "derivatives_skew_regime": "BALANCED_CHOP",
                    "derivatives_notes": "AI synthesis fallback - defensive posture default"
                }

            now_iso = datetime.now(UTC).isoformat()

            # Fast-path override if secular event catalyst is actively declared
            if active_event_playbook == "SECULAR_SHORT_BREAKDOWN":
                ai_synth["macro_bias"] = "BEARISH"
                ai_synth["market_posture"] = "TACTICAL_DEFENSE"
            elif active_event_playbook == "BULL_MOMENTUM_EXPANSION":
                ai_synth["macro_bias"] = "BULLISH"
                ai_synth["market_posture"] = "AGGRESSIVE_EXPANSION"
            
            # Derivatives Skew Circuit Breaker Guard
            is_skew_halt = bool(ai_synth.get("derivatives_skew_halt", False))
            skew_regime = ai_synth.get("derivatives_skew_regime", "BALANCED_CHOP")
            skew_notes = ai_synth.get("derivatives_notes", "")

            for db_target in ["trading_ledger.db", "trading_data.db"]:
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
                                     (json.dumps(deriv_skew.get("symbols", {})), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_summary', ?, ?)",
                                     (ai_synth.get("executive_summary", ""), now_iso))
                        conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_bias', ?, ?)",
                                     (ai_synth.get("macro_bias", "NEUTRAL_CHOP"), now_iso))
                except Exception as de:
                    log.warning(f"Error persisting macro briefing to {db_target}: {de}")

            if is_skew_halt:
                log.critical(f"[DERIVATIVES-SKEW-RADAR] ⚠️ DERIVATIVES SKEW HALT ACTIVATED! {skew_regime} | {skew_notes}")
            else:
                log.info(f"[DERIVATIVES-SKEW-RADAR] 📡 Skew Regime: {skew_regime} | Posture: {ai_synth.get('market_posture')} | Halt={is_skew_halt} | {skew_notes}")

            log.info(f"[AI-MACRO] Synthesis Complete: Posture={ai_synth.get('market_posture')} | Bias={ai_synth.get('macro_bias')} | {ai_synth.get('executive_summary')}")
        except Exception as syn_e:
            log.warning(f"[AI-MACRO] Periodic synthesis error: {syn_e}")

        # Feature 4: The Bitcoin Anchor (Cross-Market Contagion Filter)
        # Evaluates Bitcoin microstructure, funding skew, and order book absorption to protect altcoins.
        # Safety: ensure now_iso is always defined even if macro synthesis block above threw an exception.
        if 'now_iso' not in locals():
            now_iso = datetime.now(UTC).isoformat()
        try:
            btc_sym = "BTCUSDT"
            btc_fr = deriv_skew.get("symbols", {}).get(btc_sym, {}).get("funding_rate_8h", 0.0001) if 'deriv_skew' in locals() else 0.0001
            btc_oi = deriv_skew.get("symbols", {}).get(btc_sym, {}).get("open_interest_contracts", 0.0) if 'deriv_skew' in locals() else 0.0
            btc_ofi = 0.0
            if os.path.exists("microstructure_state.json"):
                try:
                    with open("microstructure_state.json", "r") as mf:
                        m_json = json.load(mf)
                        btc_ofi = float(m_json.get("symbols", {}).get("BTCUSDT", {}).get("normalized_ofi", 0.0))
                except Exception:
                    pass

            btc_anchor_packet = {
                "timestamp": now_iso,
                "btc_price": btc_price if 'btc_price' in locals() else 0.0,
                "btc_macro_trend": btc_macro_trend if 'btc_macro_trend' in locals() else "UNKNOWN",
                "warden_regime": warden_regime if 'warden_regime' in locals() else "UNKNOWN",
                "btc_funding_rate_8h": btc_fr,
                "btc_open_interest": btc_oi,
                "btc_normalized_ofi": btc_ofi,
                "spy_1h_change": data['spy_1h_change'] if data else 0.0,
                "dxy_price": data.get('dxy_price', 0.0) if data else 0.0
            }

            anchor_res = macro_loop.run_until_complete(ai_engine.evaluate_btc_anchor(btc_anchor_packet, timeout_sec=12.0))
            if anchor_res:
                btc_state = anchor_res.get("btc_anchor_state", "HEALTHY_EXPANSION")
                alt_pause = bool(anchor_res.get("alt_long_pause", False))
                risk_score = float(anchor_res.get("contagion_risk_score", 20))
                directive = anchor_res.get("anchor_directive", "Normal operations.")
                key_pivot = float(anchor_res.get("key_btc_pivot", 0.0))

                log.info(f"[BTC-ANCHOR] ⚓ BTC State: {btc_state} | AltLongPause: {alt_pause} | ContagionRisk: {risk_score}/100 | {directive}")

                for db_target in ["trading_ledger.db", "trading_data.db"]:
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
                    except Exception as de:
                        log.warning(f"Error persisting BTC anchor to {db_target}: {de}")
        except Exception as anchor_err:
            log.warning(f"[BTC-ANCHOR] Periodic Bitcoin anchor error: {anchor_err}")

        # Engine 4: AI Delta-Neutral Hedging Oracle (Macro Volatility Shield)
        try:
            # Query active positions across open_positions and berserker_open_positions
            open_pos_list = []
            net_long_usd = 0.0
            net_short_usd = 0.0
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as p_conn:
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
            posture = ai_synth.get("market_posture", "TACTICAL_DEFENSE") if 'ai_synth' in locals() else "NEUTRAL"
            
            # Fetch avoid list / weakest assets
            weakest_list = ["ETH/USDT", "XRP/USDT"]
            try:
                with sqlite3.connect("trading_data.db", timeout=5.0) as a_conn:
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
                "macro_regime": data.get('macro_regime', 'NEUTRAL') if data else "NEUTRAL",
                "macro_posture": posture,
                "spy_1h_change": data.get('spy_1h_change', 0.0) if data else 0.0,
                "dxy_price": data.get('dxy_price', 0.0) if data else 0.0,
                "weakest_fleet_assets": weakest_list
            }

            hedge_rec = macro_loop.run_until_complete(ai_engine.calculate_delta_neutral_hedge(hedging_context, timeout_sec=15.0))
            if hedge_rec:
                log.info(f"[HEDGING-ORACLE] 🛡️ Shield={hedge_rec.get('risk_shield_level')} | HedgeReq={hedge_rec.get('hedge_required')} | Asset={hedge_rec.get('hedge_asset')} ({hedge_rec.get('hedge_side')}) | Target=${hedge_rec.get('target_short_notional')} | Rationale={hedge_rec.get('rationale')}")
                for db_target in ["trading_ledger.db", "trading_data.db"]:
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

        # Feature 3: Chief Strategist Asset Hierarchy Leaderboard (Every 2 hours = 8 iterations)
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
                        with sqlite3.connect("trading_data.db", timeout=5.0) as c_m:
                            r_lbl = c_m.execute("SELECT value FROM system_state WHERE key = ?", (f"regime_{clean_s}",)).fetchone()
                            if r_lbl: asset_regime = r_lbl[0]
                            r_adx = c_m.execute("SELECT value FROM system_state WHERE key = ?", (f"adx_{clean_s}",)).fetchone()
                            if r_adx: asset_adx = float(r_adx[0])
                            r_h = c_m.execute("SELECT hurst FROM active_fleet_config WHERE symbol LIKE ?", (f"%{clean_s[:3]}%",)).fetchone()
                            if r_h and r_h[0]: asset_hurst = float(r_h[0])
                    except Exception:
                        pass
                        
                    assets_metrics[s] = {
                        "regime": asset_regime,
                        "adx": asset_adx,
                        "hurst": asset_hurst
                    }

                now_iso = datetime.now(UTC).isoformat()
                fleet_context = {
                    "timestamp": now_iso,
                    "btc_price": btc_price if 'btc_price' in locals() else 0.0,
                    "macro_posture": ai_synth.get("market_posture", "TACTICAL_DEFENSE") if 'ai_synth' in locals() else "NEUTRAL",
                    "assets": assets_metrics
                }

                ranking = macro_loop.run_until_complete(ai_engine.rank_asset_hierarchy(fleet_context, timeout_sec=20.0))
                if ranking:
                    rank_str = json.dumps(ranking)
                    leaders = ", ".join(ranking.get("priority_leaders", []))
                    avoid = ", ".join(ranking.get("avoid_chop_list", []))
                    log.info(f"[CHIEF-STRATEGIST] 👑 Asset Hierarchy Updated: Leaders=[{leaders}] | Avoid=[{avoid}] | Rationale={ranking.get('fleet_focus_rationale')}")

                    for db_target in ["trading_ledger.db", "trading_data.db"]:
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

        # Feature 6: AI 6-Hour Strategic Macro Blueprint & Key Level Roadmap
        # Generates or updates strategic macro gameplan every 6 hours (00:00, 06:00, 12:00, 18:00 UTC) or if stale (>6h)
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
                with sqlite3.connect("trading_data.db", timeout=5.0) as b_conn:
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
                
                # Fetch live prices across fleet for accurate S/R anchoring
                live_prices = {}
                if os.path.exists("microstructure_state.json"):
                    try:
                        with open("microstructure_state.json", "r") as msf:
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
                    "btc_price": live_prices.get("BTC/USDT", btc_price if 'btc_price' in locals() else 0.0),
                    "asset_current_prices": live_prices,
                    "macro_regime": data.get('macro_regime', 'NEUTRAL') if data else "NEUTRAL",
                    "spy_price": data.get('spy_price', 0.0) if data else 0.0,
                    "dxy_price": data.get('dxy_price', 0.0) if data else 0.0,
                    "fleet_symbols": fleet_symbols[:10]
                }
                daily_bp = macro_loop.run_until_complete(ai_engine.generate_daily_market_blueprint(fleet_symbols[:10], bp_snapshot, timeout_sec=25.0))
                if daily_bp:
                    daily_bp["session_tag"] = session_tag
                    daily_bp["session_name"] = session_name
                    bp_json = json.dumps(daily_bp)
                    now_iso = datetime.now(UTC).isoformat()
                    log.info(f"[MACRO-BLUEPRINT] 🗺️ 6-Hour Blueprint Generated [{session_tag}] ({daily_bp.get('macro_bias')}): {daily_bp.get('daily_narrative')[:80]}...")
                    for db_target in ["trading_ledger.db", "trading_data.db"]:
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

