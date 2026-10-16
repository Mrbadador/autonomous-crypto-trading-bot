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
    """
    Fetches Top USDT futures pairs by 24h quote volume directly from Bitget V2 Mix Market API.
    Eliminates cross-exchange basis drift.
    """
    try:
        url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json().get("data", [])
        
        usdt_pairs = [p for p in data if p.get('symbol', '').endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x.get('usdtVolume') or x.get('quoteVolume') or 0.0), reverse=True)
        
        return [p['symbol'] for p in usdt_pairs[:limit]]
    except Exception as e:
        log.error(f"Error fetching Bitget Top pairs: {e}")
        return []

def fetch_futures_ohlcv(symbol, interval="15m", limit=300):
    """
    Pulls OHLCV bars directly from Bitget V2 Mix Market API.
    Matches live execution basis with zero price wick divergence.
    """
    clean_sym = symbol.replace('/', '').replace(':USDT', '').replace('-', '').upper()
    bg_tf_map = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "1H": "1H", "4h": "4H", "4H": "4H", "6h": "6H", "6H": "6H",
        "12h": "12H", "12H": "12H", "1d": "1Dutc", "1D": "1Dutc"
    }
    granularity = bg_tf_map.get(interval, interval if interval in ("1H", "4H", "6H", "12H") else "15m")
    url = f"https://api.bitget.com/api/v2/mix/market/candles?symbol={clean_sym}&granularity={granularity}&limit={limit}&productType=USDT-FUTURES"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            log.error(f"Bitget candles API error for {symbol} ({granularity}): HTTP {response.status_code} - {response.text}")
            return None
        res_json = response.json()
        if res_json.get("code") == "00000":
            klines = res_json.get("data", [])
            if not klines:
                return None
            # Bitget format: [ts, open, high, low, close, base_vol, quote_vol]
            df = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume", "quote_asset_volume"
            ])
            # Bitget returns newest first; sort ascending for chronological indicators
            df["open_time"] = df["open_time"].astype(int)
            df = df.sort_values("open_time").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
        else:
            log.error(f"Bitget candles API returned non-success code for {symbol} ({granularity}): code={res_json.get('code')} msg={res_json.get('msg')}")
    except Exception as e:
        log.error(f"Failed to fetch Bitget historical bars for {symbol} ({granularity}): {e}")
    return None

def calculate_variance_ratio(prices, q=10):
    """
    Lo-MacKinlay (1988) Heteroskedasticity-Robust Variance Ratio Test.
    Evaluates whether prices follow a random walk, mean-revert, or trend.
    Returns (VR, z_stat):
      - z_stat < -1.645: Statistically significant mean-reversion (chop/grid regime)
      - z_stat > +1.645: Statistically significant persistence (trending regime)
      - |z_stat| <= 1.645: Pure Gaussian random walk / uninformative
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < (2 * q + 10):
        return 1.0, 0.0
    try:
        log_p = np.log(prices)
        ret_1 = np.diff(log_p)
        n = len(ret_1)
        mu = np.mean(ret_1)

        sigma_1_sq = np.sum((ret_1 - mu) ** 2) / (n - 1)
        if sigma_1_sq <= 1e-14:
            return 1.0, 0.0

        # Overlapping q-period returns
        ret_q = log_p[q:] - log_p[:-q]
        m = q * (n - q + 1) * (1.0 - (q / n))
        sigma_q_sq = np.sum((ret_q - q * mu) ** 2) / m

        vr = sigma_q_sq / sigma_1_sq

        # Heteroskedasticity-consistent asymptotic variance
        theta = 0.0
        denom = np.sum((ret_1 - mu) ** 2) ** 2
        for j in range(1, q):
            w = 2.0 * (q - j) / q
            delta_j = np.sum(((ret_1[j:] - mu) ** 2) * ((ret_1[:-j] - mu) ** 2)) / denom
            theta += (w ** 2) * delta_j

        z_stat = (vr - 1.0) / np.sqrt(theta) if theta > 1e-12 else 0.0
        return float(vr), float(z_stat)
    except Exception as e:
        log.warning(f"Error calculating Variance Ratio: {e}")
        return 1.0, 0.0

def calculate_hurst(prices):
    """
    Variance-Ratio & Anis-Lloyd Corrected Persistence Metric.
    Converts robust Lo-MacKinlay Variance Ratio into an equivalent Hurst exponent [0, 1]:
      - H = 0.50: Random walk (VR = 1.0)
      - H > 0.50: Trending persistence (VR > 1.0)
      - H < 0.50: Mean reversion / Chop (VR < 1.0)
    Free from short-sample upward Anis-Lloyd R/S bias.
    """
    if len(prices) < 30:
        return 0.50
    try:
        vr, z_stat = calculate_variance_ratio(prices, q=10)
        # Map VR to Hurst exponent: VR = 2^(2H - 1), so H = 0.5 + log2(VR)/2
        if vr > 0:
            h = 0.5 + (np.log2(vr) / 2.0)
            return float(np.clip(h, 0.05, 0.95))
        return 0.50
    except Exception as e:
        log.warning(f"Error calculating Hurst from VR: {e}")
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
            min_runway_pct = float(getattr(config, "MTF_MIN_RUNWAY_PCT", 0.004)) * 100.0  # matches config default 0.4%
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

        # 3. 4-Hour Heavy Fleet Sweep (Every 16 cycles of 15m = 4 hours)
        if cycle_count % 16 == 0:
            log.info(f"Starting 4-hour Asset Scout & Funding Rate sweep (Cycle #{cycle_count})...")
            
            # Phase 3 (V28.8) Funding Rate Yield Arbitrage Sweep
            try:
                log.info("Harvesting active 8-hour funding rates directly from Bitget V2...")
                funding_url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"
                resp = requests.get(funding_url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if isinstance(data, list):
                        funding_rates = {}
                        for item in data:
                            sym = item.get('symbol')
                            rate_str = item.get('fundingRate')
                            if sym and rate_str is not None:
                                try:
                                    funding_rates[sym] = float(rate_str)
                                except Exception:
                                    pass
                        
                        # BIS Crash-Risk Contagion Shield: Funding > +0.045% across active fleet while 4H ROC <= 0
                        fleet_clean = {s.upper().replace("/", "").replace("-", "") for s in getattr(config, "SYMBOLS", [])}
                        fleet_funding_rates = {k: v for k, v in funding_rates.items() if k.upper().replace("/", "").replace("-", "") in fleet_clean} if fleet_clean else funding_rates
                        high_funding = any(float(r) >= 0.00045 for r in fleet_funding_rates.values())
                        btc_df_4h = fetch_futures_ohlcv("BTCUSDT", interval="4h", limit=10)
                        btc_roc_negative = False
                        if btc_df_4h is not None and len(btc_df_4h) >= 2:
                            c_now = float(btc_df_4h["close"].iloc[-1])
                            c_prev = float(btc_df_4h["close"].iloc[-2])
                            btc_roc_negative = (c_now <= c_prev)

                        funding_crash_state = "CRITICAL" if (high_funding and btc_roc_negative) else "NORMAL"
                        for db_name in ["trading_data.db", "trading_ledger.db"]:
                            try:
                                with sqlite3.connect(db_name, timeout=5.0) as conn_cr:
                                    conn_cr.execute("""
                                        INSERT OR REPLACE INTO system_state (key, value, updated_at)
                                        VALUES ('funding_crash_risk', ?, ?)
                                    """, (funding_crash_state, datetime.now(timezone.utc).isoformat()))
                            except Exception:
                                pass

                        valid_opps = yield_hunter.evaluate_opportunities(funding_rates)
                        for opp in valid_opps:
                            yield_pct = opp["annualized_yield"] * 100.0
                            log.info(f"[YIELD-HUNTER] Delta-Neutral Arbitrage: {opp['symbol']} | Net Yield: {yield_pct:.2f}% | Dir: {opp['direction']}")
                else:
                    log.warning(f"Bitget tickers returned status {resp.status_code}")
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
                            
                            # P0-1 FIX: asset_scout writes advisory fleet metrics to active_fleet_config
                            # but does NOT overwrite ALLOWED_SYMBOLS in system_state (which is config-owned).
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
