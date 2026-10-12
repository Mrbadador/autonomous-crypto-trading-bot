import sqlite3
import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, UTC, timezone, timedelta
import dns_fix  # noqa: F401
import config
import ccxt
import asyncio
import json
import ai_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] ASSET_SCOUT: %(message)s")
log = logging.getLogger("asset_scout")

from market_neutral_agents import FundingArbitrageur
yield_hunter = FundingArbitrageur(capital=500.0, min_annual_yield=0.08)


def init_db(db_path="trading_data.db"):
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_fleet_config (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    hurst REAL,
                    updated_at TEXT
                )
            """)
        conn.close()
    except Exception as e:
        log.critical(f"Failed to init active_fleet_config table: {e}")
        raise

def get_top_futures_pairs(limit=50):
    """Fetches Top USDT futures pairs by 24h quote volume."""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        usdt_pairs = [p for p in data if p['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
        
        return [p['symbol'] for p in usdt_pairs[:limit]]
    except Exception as e:
        log.error(f"Error fetching Top pairs: {e}")
        return []

def fetch_futures_ohlcv(symbol, interval="15m", limit=105):
    """Pulls OHLCV bars from Binance Futures."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            klines = response.json()
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
    except Exception as e:
        log.error(f"Failed to fetch historical bars for {symbol}: {e}")
    return None

def calculate_hurst(prices):
    """Estimates Hurst Exponent over prices using Rescaled Range (R/S) Analysis."""
    if len(prices) < 50:
        return 0.50
    try:
        returns = np.diff(np.log(prices))
        N = len(returns)
        lags = [10, 20, 30, 40, 50]
        RS_values = []
        for lag in lags:
            num_subsets = N // lag
            if num_subsets == 0:
                continue
            rs_sub = []
            for i in range(num_subsets):
                sub = returns[i*lag : (i+1)*lag]
                mean_adj = sub - np.mean(sub)
                cum_sum = np.cumsum(mean_adj)
                r = np.max(cum_sum) - np.min(cum_sum)
                s = np.std(sub)
                if s > 0:
                    rs_sub.append(r / s)
            if rs_sub:
                RS_values.append(np.mean(rs_sub))
        if len(RS_values) < 2:
            return 0.50
        poly = np.polyfit(np.log(lags[:len(RS_values)]), np.log(RS_values), 1)
        return float(poly[0])
    except Exception as e:
        return 0.50

def run_ai_liquidity_scan():
    """
    Feature 1: 15-Minute Multi-Asset Institutional Liquidity & SFP Radar.
    Pulls 15m OHLCV bars across fleet assets, sends 1 batch request to Gemini Flash Lite,
    writes active patterns to trading_ledger.db (active_patterns) and telemetry to system_state (ai_liquidity_radar).
    """
    try:
        log.info("[AI-RADAR] Commencing 15-minute Multi-Asset Institutional Liquidity & SFP Scan...")
        fleet_symbols = getattr(config, "SYMBOLS", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT'])
        
        batch = {}
        for s in fleet_symbols[:10]:
            clean_s = s.replace('/', '').replace(':USDT', '').replace('-', '').upper()
            df = fetch_futures_ohlcv(clean_s, interval="15m", limit=35)
            if df is not None and len(df) >= 20:
                df["volume"] = df["volume"].astype(float)
                c = df["close"].values
                h = df["high"].values
                l = df["low"].values
                v = df["volume"].values
                
                last_close = float(c[-1])
                swing_high = float(np.max(h[-15:]))
                swing_low = float(np.min(l[-15:]))
                vol_mean = float(np.mean(v[-15:]))
                vol_ratio = round(float(v[-1]) / max(1e-5, vol_mean), 2)
                
                recent = []
                for i in range(-3, 0):
                    recent.append({
                        "o": round(float(df["open"].iloc[i]), 4),
                        "h": round(float(df["high"].iloc[i]), 4),
                        "l": round(float(df["low"].iloc[i]), 4),
                        "c": round(float(df["close"].iloc[i]), 4),
                        "v": round(float(df["volume"].iloc[i]), 2),
                    })
                
                batch[s] = {
                    "current_price": last_close,
                    "swing_high_15m": swing_high,
                    "swing_low_15m": swing_low,
                    "dist_to_swing_high_pct": round(((swing_high - last_close) / last_close) * 100, 2),
                    "dist_to_swing_low_pct": round(((last_close - swing_low) / last_close) * 100, 2),
                    "volume_surge_ratio": vol_ratio,
                    "last_3_candles": recent
                }
                
        if not batch:
            log.warning("[AI-RADAR] No valid candle bars fetched for fleet symbols. Skipping scan cycle.")
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        packet = {
            "timestamp": now_iso,
            "fleet": batch
        }
        
        # Async invocation of AI scanner
        res = asyncio.run(ai_engine.scan_multi_asset_liquidity_patterns(packet, timeout_sec=20.0))
        detected_patterns = res.get("detected_patterns", [])
        log.info(f"[AI-RADAR] AI Scan Complete: {len(detected_patterns)} patterns detected. Summary: {res.get('market_summary')}")
        
        # 1. Write detected patterns into trading_ledger.db (active_patterns table)
        try:
            conn_ledger = sqlite3.connect("trading_ledger.db", timeout=30.0)
            conn_ledger.execute("PRAGMA journal_mode=WAL;")
            conn_ledger.execute("PRAGMA busy_timeout=30000;")
            with conn_ledger:
                conn_ledger.execute("""
                    CREATE TABLE IF NOT EXISTS active_patterns (
                        timestamp TEXT,
                        symbol TEXT,
                        timeframe TEXT,
                        pattern_name TEXT,
                        context_score REAL
                    )
                """)
                cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                conn_ledger.execute("DELETE FROM active_patterns WHERE timestamp < ?", (cutoff,))
                
                for pat in detected_patterns:
                    sym = pat.get("symbol", "")
                    pat_type = pat.get("pattern_type", "LIQUIDITY_SWEEP")
                    tf = pat.get("timeframe", "15m")
                    score = float(pat.get("quality_score", 80.0))
                    
                    sym_clean = sym.replace('/', '').replace(':', '').replace('-', '').upper()
                    conn_ledger.execute("""
                        INSERT INTO active_patterns (timestamp, symbol, timeframe, pattern_name, context_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (now_iso, sym_clean, tf, pat_type, score))
                    
                    if '/' in sym:
                        conn_ledger.execute("""
                            INSERT INTO active_patterns (timestamp, symbol, timeframe, pattern_name, context_score)
                            VALUES (?, ?, ?, ?, ?)
                        """, (now_iso, sym, tf, pat_type, score))
            conn_ledger.close()
        except Exception as le:
            log.error(f"[AI-RADAR] Error saving to active_patterns in trading_ledger.db: {le}")
            
        # 2. Write full telemetry into system_state for Streamlit HUD & agents
        for db_name in ["trading_data.db", "trading_ledger.db"]:
            try:
                conn = sqlite3.connect(db_name, timeout=20.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=20000;")
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                        VALUES ('ai_liquidity_radar', ?, ?)
                    """, (json.dumps(res), now_iso))
                    conn.execute("""
                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                        VALUES ('current_status_asset_scout', 'ONLINE', ?)
                    """, (now_iso,))
                conn.close()
            except Exception as se:
                log.warning(f"[AI-RADAR] Error saving ai_liquidity_radar to {db_name}: {se}")
                
    except Exception as e:
        log.error(f"[AI-RADAR] Error during AI liquidity scan: {e}", exc_info=True)


def run_ai_mtf_confluence_scan():
    """
    Engine 3: 15-Minute Multi-Timeframe (15m + 1h + 4h) Confluence Gate.
    Fetches 15m, 1h, and 4h OHLCV across fleet assets, builds MTF structural summary,
    and calls ai_engine.evaluate_mtf_confluence (dispatched via Gemini Flash Lite).
    Writes output to system_state ('ai_mtf_confluence').
    """
    try:
        log.info("[MTF-CONFLUENCE] Commencing 15-minute Multi-Timeframe (15m/1h/4h) Confluence Scan...")
        fleet_symbols = getattr(config, "SYMBOLS", ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'NEAR/USDT'])

        batch_mtf = {}
        for s in fleet_symbols[:10]:
            clean_s = s.replace('/', '').replace(':USDT', '').replace('-', '').upper()
            df_15m = fetch_futures_ohlcv(clean_s, interval="15m", limit=30)
            df_1h = fetch_futures_ohlcv(clean_s, interval="1h", limit=30)
            df_4h = fetch_futures_ohlcv(clean_s, interval="4h", limit=30)

            if df_15m is not None and len(df_15m) >= 20 and df_1h is not None and len(df_1h) >= 15:
                df_15m["volume"] = df_15m["volume"].astype(float)
                # 15m metrics
                c_15m = df_15m["close"].values
                curr_p = max(1e-5, float(c_15m[-1]))
                ema20_15m = float(pd.Series(c_15m).ewm(span=20).mean().iloc[-1])
                ema50_15m = float(pd.Series(c_15m).ewm(span=50).mean().iloc[-1])

                # 1h metrics
                c_1h = df_1h["close"].values
                h_1h = df_1h["high"].values
                l_1h = df_1h["low"].values
                ema20_1h = float(pd.Series(c_1h).ewm(span=20).mean().iloc[-1])
                swing_h_1h = float(np.max(h_1h[-15:]))
                swing_l_1h = float(np.min(l_1h[-15:]))

                # 4h metrics
                h_4h = df_4h["high"].values if df_4h is not None and len(df_4h) >= 10 else h_1h
                l_4h = df_4h["low"].values if df_4h is not None and len(df_4h) >= 10 else l_1h
                c_4h = df_4h["close"].values if df_4h is not None and len(df_4h) >= 10 else c_1h
                ema50_4h = float(pd.Series(c_4h).ewm(span=min(50, len(c_4h))).mean().iloc[-1])
                major_res_4h = float(np.max(h_4h[-15:]))
                major_sup_4h = float(np.min(l_4h[-15:]))

                batch_mtf[s] = {
                    "current_price": curr_p,
                    "15m": {
                        "trend": "BULLISH" if curr_p > ema20_15m > ema50_15m else ("BEARISH" if curr_p < ema20_15m < ema50_15m else "CHOP"),
                        "ema20": round(ema20_15m, 4),
                        "ema50": round(ema50_15m, 4)
                    },
                    "1h": {
                        "trend": "BULLISH" if curr_p > ema20_1h else "BEARISH",
                        "swing_high": swing_h_1h,
                        "swing_low": swing_l_1h,
                        "dist_to_1h_high_pct": round(((swing_h_1h - curr_p) / curr_p) * 100.0, 2),
                        "dist_to_1h_low_pct": round(((curr_p - swing_l_1h) / curr_p) * 100.0, 2)
                    },
                    "4h": {
                        "major_resistance": major_res_4h,
                        "major_support": major_sup_4h,
                        "ema50": round(ema50_4h, 4),
                        "dist_to_major_resistance_pct": round(((major_res_4h - curr_p) / curr_p) * 100.0, 2),
                        "dist_to_major_support_pct": round(((curr_p - major_sup_4h) / curr_p) * 100.0, 2)
                    }
                }

        if not batch_mtf:
            log.warning("[MTF-CONFLUENCE] Could not assemble fleet MTF matrix. Skipping cycle.")
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        mtf_packet = {
            "timestamp": now_iso,
            "assets": batch_mtf
        }

        res = asyncio.run(ai_engine.evaluate_mtf_confluence(mtf_packet, timeout_sec=25.0))
        if res:
            # Deterministic quantitative enforcement on LLM output:
            min_runway_pct = float(getattr(config, "MTF_MIN_RUNWAY_PCT", 0.010)) * 100.0  # e.g. 1.0%
            c_matrix = res.get("confluence_matrix", {})
            for s, entry in c_matrix.items():
                m_info = batch_mtf.get(s)
                if not m_info or "4h" not in m_info:
                    for bk, bv in batch_mtf.items():
                        if bk.replace("/", "").upper() == s.replace("/", "").replace(":USDT", "").upper():
                            m_info = bv
                            break
                if m_info and "4h" in m_info:
                    dist_res = float(m_info["4h"].get("dist_to_major_resistance_pct", 999.0))
                    dist_sup = float(m_info["4h"].get("dist_to_major_support_pct", 999.0))
                    res_price = m_info["4h"].get("major_resistance", 0.0)
                    sup_price = m_info["4h"].get("major_support", 0.0)
                    if not entry.get("key_htf_barrier") or float(entry.get("key_htf_barrier", 0.0)) <= 0:
                        entry["key_htf_barrier"] = res_price if dist_res < dist_sup else sup_price

                    # If within min_runway_pct of 4h resistance wall, MUST veto long
                    if 0.0 <= dist_res < min_runway_pct:
                        entry["veto_long"] = True
                        entry["alignment"] = "HTF_RESISTANCE_IMMINENT"
                        entry["structural_rationale"] = (
                            f"Deterministic Guard: Price is only {dist_res:.2f}% below 4h resistance "
                            f"(${res_price}). Long strictly vetoed (Runway < {min_runway_pct:.1f}%)."
                        )
                    # If within min_runway_pct of 4h support wall, MUST veto short
                    if 0.0 <= dist_sup < min_runway_pct:
                        entry["veto_short"] = True
                        entry["alignment"] = "HTF_SUPPORT_IMMINENT"
                        entry["structural_rationale"] = (
                            f"Deterministic Guard: Price is only {dist_sup:.2f}% above 4h support "
                            f"(${sup_price}). Short strictly vetoed (Runway < {min_runway_pct:.1f}%)."
                        )

            log.info(f"[MTF-CONFLUENCE] Scan Complete. Fleet Bias: {res.get('fleet_bias')} | Summary: {res.get('market_summary')}")
            for db_name in ["trading_data.db", "trading_ledger.db"]:
                try:
                    conn = sqlite3.connect(db_name, timeout=20.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA busy_timeout=20000;")
                    with conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO system_state (key, value, updated_at)
                            VALUES ('ai_mtf_confluence', ?, ?)
                        """, (json.dumps(res), now_iso))
                    conn.close()
                except Exception as de:
                    log.warning(f"[MTF-CONFLUENCE] Error saving to {db_name}: {de}")
    except Exception as e:
        log.error(f"[MTF-CONFLUENCE] Error during MTF scan: {e}", exc_info=True)


def run_scout():
    init_db()
    cycle_count = 0
    while True:
        # 1. Real-Time 15-Minute Institutional Liquidity & SFP Radar Sweep
        run_ai_liquidity_scan()

        # 2. Real-Time 15-Minute Multi-Timeframe Confluence Gate (15m + 1h + 4h)
        run_ai_mtf_confluence_scan()

        # 2. 4-Hour Heavy Fleet Sweep (Every 16 cycles of 15m)
        if cycle_count % 16 == 0:
            log.info(f"Starting 4-hour Asset Scout & Funding Rate sweep (Cycle #{cycle_count})...")
            
            # Phase 3 (V28.8) Funding Rate Yield Arbitrage Sweep
            try:
                log.info("Harvesting active 8-hour funding rates from Binance...")
                funding_url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                resp = requests.get(funding_url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        funding_rates = {}
                        for item in data:
                            sym = item.get('symbol')
                            rate_str = item.get('lastFundingRate')
                            if sym and rate_str is not None:
                                funding_rates[sym] = float(rate_str)
                        
                        valid_opps = yield_hunter.evaluate_opportunities(funding_rates)
                        for opp in valid_opps:
                            yield_pct = opp["annualized_yield"] * 100.0
                            log.info(f"[YIELD-HUNTER] Delta-Neutral Arbitrage: {opp['symbol']} | Net Yield: {yield_pct:.2f}% | Dir: {opp['direction']}")
                else:
                    log.warning(f"Binance premiumIndex returned status {resp.status_code}")
            except Exception as e:
                log.error(f"Failed to execute yield hunter sweep: {e}")

            # Load exchange markets to filter valid symbols
            markets = {}
            try:
                from exchange_factory import create_sync_exchange
                exchange = create_sync_exchange()
                markets = exchange.load_markets()
            except Exception as me:
                log.error(f"Failed to load markets from exchange: {me}")

            top_pairs = get_top_futures_pairs(50)
            
            results = []
            for sym in top_pairs:
                df = fetch_futures_ohlcv(sym)
                if df is not None and len(df) >= 50:
                    hurst = calculate_hurst(df['close'].values)
                    results.append({'symbol': sym, 'hurst': hurst})
                time.sleep(0.1)
                
            if results:
                results.sort(key=lambda x: x['hurst'], reverse=True)
                
                # V28.0 Strict Exclusion List
                exclusion_list = [
                    'XAU', 'XAG', 'PAXG', 'EUR', 'GBP', 'TRY', 'ZAR', 'RUB', 'BRL', 'JPY', 
                    'UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT', 'USDC', 'FDUSD', 'TUSD', 'USDP'
                ]
                
                # V28.0 Crypto-Lock Filtering: Exclude non-ASCII, forbidden, and invalid exchange assets
                clean_crypto_assets = []
                for r in results:
                    base_symbol = str(r['symbol']).replace('/USDT', '').replace('USDT', '').strip().upper()
                    if not base_symbol.isascii():
                        continue
                    if base_symbol in exclusion_list:
                        continue
                    if markets and (base_symbol + '/USDT:USDT') not in markets and (base_symbol + '/USDT') not in markets:
                        continue
                    clean_crypto_assets.append(r)
                
                top_10 = clean_crypto_assets[:10]
                timestamp = datetime.now(UTC).isoformat()
                
                attempts = 3
                success = False
                while attempts > 0 and not success:
                    try:
                        conn = sqlite3.connect("trading_data.db", timeout=60.0)
                        conn.execute("PRAGMA journal_mode=WAL;")
                        conn.execute("PRAGMA busy_timeout=60000;")
                        with conn:
                            formatted_symbols = [p['symbol'][:-4] + '/USDT' for p in top_10]
                            for idx, p in enumerate(top_10):
                                formatted_symbol = p['symbol'][:-4] + '/USDT'
                                conn.execute("""
                                    INSERT OR REPLACE INTO active_fleet_config (id, symbol, hurst, updated_at)
                                    VALUES (?, ?, ?, ?)
                                """, (idx+1, formatted_symbol, p['hurst'], timestamp))
                            
                            conn.execute("DELETE FROM active_fleet_config WHERE id > ?", (len(top_10),))
                            
                            # Connect directly to ALLOWED_SYMBOLS gate in conviction_filter.py
                            import json as _json
                            conn.execute("""
                                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                VALUES ('ALLOWED_SYMBOLS', ?, ?)
                            """, (_json.dumps(formatted_symbols), timestamp))

                            conn.execute("""
                                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                VALUES ('asset_scout_last_sweep_timestamp', ?, ?)
                            """, (timestamp, timestamp))
                            conn.execute("""
                                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                VALUES ('asset_scout_fleet_size', ?, ?)
                            """, (str(len(top_10)), timestamp))
                            conn.execute("""
                                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                VALUES ('current_status_asset_scout', 'ONLINE', ?)
                            """, (timestamp,))
                            
                        formatted_names = [p['symbol'][:-4] + '/USDT' for p in top_10]
                        log.info(f"Successfully updated active_fleet_config. Top {len(top_10)}: {formatted_names}")
                        success = True
                    except Exception as e:
                        attempts -= 1
                        log.warning(f"Database lock detected writing active fleet config. Retrying in 2s... ({attempts} attempts left): {e}")
                        if attempts > 0:
                            time.sleep(2.0)
                        else:
                            log.error(f"Failed to update active fleet config database after 3 attempts: {e}")
            else:
                try:
                    conn = sqlite3.connect("trading_data.db", timeout=10.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA busy_timeout=10000;")
                    with conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO system_state (key, value, updated_at)
                            VALUES ('current_status_asset_scout', 'OFFLINE', ?)
                        """, (datetime.now(UTC).isoformat(),))
                    conn.close()
                except Exception:
                    pass
                    
        cycle_count += 1
        log.info(f"[ASSET_SCOUT] Cycle #{cycle_count} complete. Sleeping 15 minutes until next AI Liquidity sweep...")
        time.sleep(15 * 60)

if __name__ == "__main__":
    run_scout()
