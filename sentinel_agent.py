# sentinel_agent.py — V22.0 Async Sentiment Sentinel (Zero-API-Key RSS News Auditor)
#
# V22.0 Performance Upgrade:
# - fetch_latest_headlines() is now fully async (aiohttp, non-blocking).
# - A background cache is populated every CACHE_TTL_SECONDS so the critical
#   order path only reads an in-memory flag — zero network latency on hot path.
# - perform_disaster_audit() now returns instantly from cache (< 1ms).
# - async_start_background_refresh() must be called once during bot startup.
#
# Dependencies: aiohttp (already used by the rest of the engine)

import asyncio
import aiohttp
import socket
import logging
import sqlite3
import json
import hashlib
import os
import tempfile
from datetime import datetime, UTC, timedelta

log = logging.getLogger("sentinel_agent")

CACHE_TTL_SECONDS = 60          # Re-fetch RSS every 60 seconds in background
MAX_CACHE_AGE_SECONDS = 7200    # 2 hours max age before failing closed (VETO/BLOCK)
RSS_FETCH_TIMEOUT = 8           # Max seconds to wait for a single feed (non-blocking)
SENTIMENT_STATE_FILE = "sentiment_state.json"
FLEET_TICKERS = [
    "BTC", "ETH", "SOL", "LINK", "AVAX", "XRP",
    "ADA", "BNB", "NEAR", "DOGE", "SUI", "APT"
]


class SentimentSentinel:
    """
    V43.0 Unified Sentiment Sentinel & Fleet Kill-Switch Writer.

    - Periodically fetches RSS feeds + Fear & Greed Index and updates both
      in-memory cache and shared `sentiment_state.json` for all fleet tickers + GLOBAL.
    - Enforces a strict 2-hour freshness window (MAX_CACHE_AGE_SECONDS = 7200s):
      stale or missing news data fails CLOSED (`VETO` / `BLOCK`).
    - Supports three tiers:
      * `VETO` / `BLOCK` -> abort trade (0.0x size)
      * `WARNING` -> caution tier (0.5x size-down)
      * `CLEAR` / `SAFE` -> normal execution (1.0x size)
    """

    def __init__(self):
        log.info("[SENTINEL] V43.0 Initialized. Unified RSS + sentiment_state.json mode active.")

        self.news_feeds = [
            "https://cointelegraph.com/rss",
            "https://cointelegraph.com/rss/tag/bitcoin",
            "https://cointelegraph.com/rss/tag/solana",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ]

        self.disaster_keywords = [
            "hacked", "exploit", "exploited", "sec sues", "arrested", "indicted",
            "bankrupt", "bankruptcy", "delist", "delisting", "delisted",
            "rug pull", "fraudulent", "insolvent", "insolvency", "exchange shutdown",
            "withdrawals suspended", "trading halted", "assets seized", "sanctions",
            "nationwide crypto ban", "emergency halt",
            "ponzi", "depeg", "depegged", "cease and desist"
        ]

        self.warning_keywords = [
            "outflow", "outflows", "lawsuit", "probe", "investigation",
            "delay", "delayed", "downgrade", "sell-off", "selloff",
            "whale dump", "token unlock", "vulnerability", "congestion",
            "outage", "liquidation", "liquidations", "liquidated",
            "emergency patch", "subpoena", "hack", "war"
        ]

        self.systemic_keywords = [
            "crypto market collapse", "flash crash today",
            "tether depeg", "usdt depeg", "usdc depeg", "stablecoin depeg",
            "binance bankrupt", "binance insolvent", "binance halt", "binance seized",
            "coinbase halt", "emergency exchange halt", "nationwide crypto ban", "us crypto ban",
            "sec emergency halt", "world war", "nuclear strike", "market contagion", "systemic failure"
        ]

        self.retrospective_markers = [
            "one year after", "one-year after", "year after", "years after",
            "anniversary", "looking back", "history of", "months after",
            "remembering", "wake-up call"
        ]

        # In-memory headline cache: {ticker_upper: (verdict, message, fetched_at)}
        self._cache: dict[str, tuple[str, str, datetime]] = {}
        self._refresh_task: asyncio.Task | None = None
        self._fear_greed_raw: float = 50.0   # 0-100 scale (matches live_oracle / dashboard)
        self._fear_greed_norm: float = 0.50  # 0.0-1.0 scale (legacy sentiment_state.json field)
        self._last_fng_fetch: datetime | None = None

        # V40.0 Event-Driven Catalyst State & Headline Cache
        self._evaluated_headline_hashes: set[str] = set()
        self._active_event_playbook: str = "NORMAL_OPERATIONS"
        self._active_event_summary: str = ""
        self._active_event_horizon: str = "NEUTRAL"
        self._active_event_data: dict = {}
        self._load_cached_event_state()
        self._load_shared_sentiment_state()

    def _load_cached_event_state(self):
        """Loads most recent event catalyst state from SQLite if available."""
        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                row = conn.execute("SELECT value FROM system_state WHERE key = 'ai_event_catalyst'").fetchone()
                if row and row[0]:
                    data = json.loads(row[0])
                    updated_at_str = data.get("updated_at")
                    is_expired = False
                    if updated_at_str:
                        try:
                            updated_dt = datetime.fromisoformat(updated_at_str)
                            if updated_dt.tzinfo is None:
                                updated_dt = updated_dt.replace(tzinfo=UTC)
                            if (datetime.now(UTC) - updated_dt).total_seconds() > (4 * 3600):
                                is_expired = True
                        except Exception:
                            pass
                    if not is_expired:
                        self._active_event_data = data
                        self._active_event_playbook = data.get("tactical_playbook", "NORMAL_OPERATIONS")
                        self._active_event_summary = data.get("headline_summary", "")
                        self._active_event_horizon = data.get("expected_horizon", "NEUTRAL")
                    else:
                        self._active_event_playbook = "NORMAL_OPERATIONS"
                        self._active_event_summary = ""
                        self._active_event_horizon = "NEUTRAL"
        except Exception:
            pass

    def _load_shared_sentiment_state(self) -> bool:
        """
        Loads `sentiment_state.json` into `self._cache` ONLY if `last_update` is within
        MAX_CACHE_AGE_SECONDS (2 hours). Returns True if fresh state was loaded.
        """
        if not os.path.exists(SENTIMENT_STATE_FILE):
            return False
        try:
            with open(SENTIMENT_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            last_update_str = data.get("last_update")
            if not last_update_str:
                return False
            updated_dt = datetime.fromisoformat(last_update_str)
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=UTC)
            age_s = (datetime.now(UTC) - updated_dt).total_seconds()
            if age_s > MAX_CACHE_AGE_SECONDS:
                log.warning(f"[SENTINEL] Shared {SENTIMENT_STATE_FILE} is stale (age={age_s/3600:.1f}h > 2.0h). Ignoring stale state.")
                return False

            if "fear_greed_raw" in data:
                self._fear_greed_raw = float(data["fear_greed_raw"])
                self._fear_greed_norm = round(self._fear_greed_raw / 100.0, 4)
            elif "fear_greed_index" in data:
                fg_val = float(data["fear_greed_index"])
                if fg_val <= 1.0:
                    self._fear_greed_norm = fg_val
                    self._fear_greed_raw = round(fg_val * 100.0, 1)
                else:
                    self._fear_greed_raw = fg_val
                    self._fear_greed_norm = round(fg_val / 100.0, 4)

            loaded_any = False
            for key, val in data.items():
                if isinstance(val, dict) and "status" in val:
                    raw_status = str(val.get("status", "SAFE")).upper()
                    verdict = str(val.get("verdict", "")).upper()
                    if not verdict:
                        if raw_status == "BLOCK":
                            verdict = "VETO"
                        elif raw_status == "WARNING":
                            verdict = "WARNING"
                        else:
                            verdict = "CLEAR"
                    reason = str(val.get("reason", f"Loaded from {SENTIMENT_STATE_FILE}"))
                    self._cache[key.upper()] = (verdict, reason, updated_dt)
                    loaded_any = True
            return loaded_any
        except Exception as e:
            log.warning(f"[SENTINEL] Failed reading {SENTIMENT_STATE_FILE}: {e}")
            return False

    def _persist_shared_state_sync(self, state_payload: dict):
        """Atomically writes `sentiment_state.json` and syncs SQLite `system_state`."""
        try:
            dir_name = os.path.dirname(os.path.abspath(SENTIMENT_STATE_FILE)) or "."
            fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp", text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state_payload, f, indent=4)
            os.replace(temp_path, SENTIMENT_STATE_FILE)
        except Exception as e:
            log.error(f"[SENTINEL] Atomic write to {SENTIMENT_STATE_FILE} failed: {e}")

        try:
            now_iso = state_payload.get("last_update", datetime.now(UTC).isoformat())
            with sqlite3.connect("trading_data.db", timeout=10.0) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('sentinel_news_cache', ?, ?)",
                    (json.dumps(state_payload), now_iso)
                )
                conn.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('fear_greed_index', ?, ?)",
                    (str(round(self._fear_greed_raw, 1)), now_iso)
                )
                conn.commit()
        except Exception as e:
            log.debug(f"[SENTINEL] SQLite state persistence warning: {e}")

    async def _fetch_fear_greed_async(self):
        """Fetches alternative.me Fear & Greed Index (cached for 15 minutes)."""
        now_utc = datetime.now(UTC)
        if self._last_fng_fetch and (now_utc - self._last_fng_fetch).total_seconds() < 900:
            return
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get("https://api.alternative.me/fng/") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        val_raw = float(data["data"][0]["value"])
                        self._fear_greed_raw = round(val_raw, 1)
                        self._fear_greed_norm = round(val_raw / 100.0, 4)
                        self._last_fng_fetch = now_utc
        except Exception as e:
            log.debug(f"[SENTINEL] Fear & Greed fetch note: {e}")

    # -----------------------------------------------------------------------
    # Background Refresh Loop
    # -----------------------------------------------------------------------
    def start_background_refresh(self, tickers: list[str] | None = None):
        """
        Launches a persistent background coroutine that refreshes headlines
        for all monitored tickers every CACHE_TTL_SECONDS.
        Call once during bot startup (after the event loop is running).
        """
        if self._refresh_task and not self._refresh_task.done():
            return  # Already running
        merged_tickers = list(dict.fromkeys((tickers or []) + FLEET_TICKERS))
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(merged_tickers), name="sentinel_refresh"
        )
        log.info(f"[SENTINEL] Background refresh started for tickers: {merged_tickers}")

    async def refresh_once(self, tickers: list[str] | None = None) -> bool:
        """
        Performs a single live refresh cycle:
        1. Fetches global RSS feeds & Fear/Greed.
        2. If 0 headlines returned, does NOT update `last_update` so dead feeds age out and fail-closed.
        3. Audits GLOBAL and all fleet tickers (`VETO`/`BLOCK`, `WARNING`, `CLEAR`/`SAFE`).
        4. Atomically writes `sentiment_state.json` and updates SQLite.
        """
        merged_tickers = list(dict.fromkeys([t.upper() for t in (tickers or [])] + FLEET_TICKERS))
        headlines = await self._fetch_all_headlines_global()
        if not headlines:
            log.warning("[SENTINEL] 0 headlines fetched from RSS feeds — keeping previous timestamp so stale cache ages out fail-closed.")
            return False

        await self._fetch_fear_greed_async()
        now_utc = datetime.now(UTC)
        now_iso = now_utc.isoformat()

        g_verdict, g_msg = self._audit_global_headlines(headlines)
        self._cache["GLOBAL"] = (g_verdict, g_msg, now_utc)

        def _entry_dict(verdict: str, reason: str) -> dict:
            if verdict == "VETO":
                return {"status": "BLOCK", "verdict": "VETO", "score": 2, "size_multiplier": 0.0, "reason": reason}
            elif verdict == "WARNING":
                return {"status": "WARNING", "verdict": "WARNING", "score": 4, "size_multiplier": 0.5, "reason": reason}
            return {"status": "SAFE", "verdict": "CLEAR", "score": 8, "size_multiplier": 1.0, "reason": reason}

        state_payload: dict = {
            "GLOBAL": _entry_dict(g_verdict, g_msg)
        }

        for ticker in merged_tickers:
            t_up = ticker.upper()
            t_verdict, t_msg = self._audit_ticker_headlines(t_up, headlines)
            # If GLOBAL is VETO, propagate BLOCK to individual tickers unless overridden
            if g_verdict == "VETO":
                self._cache[t_up] = ("VETO", f"[GLOBAL SYSTEMIC VETO] {g_msg}", now_utc)
                state_payload[t_up] = _entry_dict("VETO", f"[GLOBAL SYSTEMIC VETO] {g_msg}")
            else:
                self._cache[t_up] = (t_verdict, t_msg, now_utc)
                state_payload[t_up] = _entry_dict(t_verdict, t_msg)

        state_payload["fear_greed_index"] = self._fear_greed_norm
        state_payload["fear_greed_raw"] = self._fear_greed_raw
        state_payload["headlines_scanned"] = len(headlines)
        state_payload["active_event_playbook"] = self._active_event_playbook
        state_payload["last_update"] = now_iso

        await asyncio.to_thread(self._persist_shared_state_sync, state_payload)
        log.debug(f"[SENTINEL] Refreshed {len(merged_tickers)} tickers + GLOBAL ({len(headlines)} headlines, F&G={self._fear_greed_raw}).")

        await self.evaluate_headlines_for_catalysts(headlines)
        return True

    async def _refresh_loop(self, tickers: list[str]):
        """Runs forever, refreshing cache and `sentiment_state.json` once per cycle."""
        while True:
            try:
                await self.refresh_once(tickers)
            except Exception as e:
                log.warning(f"[SENTINEL] Background refresh cycle failed: {e}")
            await asyncio.sleep(CACHE_TTL_SECONDS)

    async def evaluate_headlines_for_catalysts(self, headlines: list[str]) -> dict:
        """
        V40.0 Evaluates fresh breaking headlines using ai_engine.evaluate_event_catalyst.
        Synthesizes breaking news with live Bitget derivatives & macro telemetry.
        """
        import re
        catalyst_keywords = [
            "war", "strike", "missile", "military", "sanction", "sanctions",
            "sec", "etf", "interest rate", "inflation", "cpi", "fomc", "fed",
            "bankrupt", "insolvent", "insolvency", "halt", "emergency",
            "depeg", "depegged", "liquidation", "ban", "banned"
        ]

        target_headline = None
        for h in headlines:
            h_clean = h.strip()
            h_lower = h_clean.lower()
            if any(m in h_lower for m in self.retrospective_markers):
                continue
            if any(re.search(rf"\b{re.escape(kw)}\b", h_lower) for kw in catalyst_keywords):
                h_hash = hashlib.md5(h_clean.encode("utf-8")).hexdigest()
                if h_hash not in self._evaluated_headline_hashes:
                    target_headline = h_clean
                    self._evaluated_headline_hashes.add(h_hash)
                    break

        if not target_headline:
            return self._active_event_data

        log.info(f"[SENTINEL-EVENT] ⚡ High-urgency catalyst detected: '{target_headline}'")

        # Gather real-time derivatives & macro context
        btc_15m_vel = 0.0
        spy_1h_chg = 0.0
        dxy_idx = 104.0
        fr_8h = 0.0001
        oi_delta = 0.0

        try:
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                row_m = conn.execute("SELECT spy_1h_change, dxy_price FROM macro_market_context ORDER BY id DESC LIMIT 1").fetchone()
                if row_m:
                    spy_1h_chg = float(row_m[0] or 0.0)
                    dxy_idx = float(row_m[1] or 104.0)

                row_fr = conn.execute("SELECT value FROM system_state WHERE key = 'derivatives_funding_rates'").fetchone()
                if row_fr and row_fr[0]:
                    fr_map = json.loads(row_fr[0])
                    fr_8h = float(fr_map.get("BTCUSDT", {}).get("funding_rate_8h", 0.0001))
        except Exception as ctx_e:
            log.debug(f"[SENTINEL-EVENT-CTX] Context fetch note: {ctx_e}")

        if os.path.exists("microstructure_state.json"):
            try:
                with open("microstructure_state.json", "r") as mf:
                    m_data = json.load(mf)
                    sym_info = m_data.get("symbols", {}).get("BTCUSDT", {})
                    btc_15m_vel = float(sym_info.get("price_change_15m_pct", 0.0))
            except Exception:
                pass

        event_packet = {
            "headline": target_headline,
            "timestamp": datetime.now(UTC).isoformat(),
            "btc_15m_velocity_pct": btc_15m_vel,
            "spy_1h_change_pct": spy_1h_chg,
            "dxy_index": dxy_idx,
            "bitget_funding_rate_8h": fr_8h,
            "bitget_open_interest_delta_usd": oi_delta,
            "recent_price_reaction": f"BTC 15m change: {btc_15m_vel:+.2f}%, SPY 1h change: {spy_1h_chg:+.2f}%"
        }

        try:
            import ai_engine
            res = await ai_engine.evaluate_event_catalyst(event_packet, timeout_sec=12.0)
            if res and isinstance(res, dict):
                playbook = res.get("tactical_playbook", "IGNORE_NOISE")
                horizon = res.get("expected_horizon", "NOISE_FUD")
                summary = res.get("headline_summary", target_headline)
                category = res.get("event_category", "NOISE_RUMOR")

                self._active_event_playbook = playbook
                self._active_event_horizon = horizon
                self._active_event_summary = summary
                self._active_event_data = res

                log.info(f"[SENTINEL-EVENT] 🎯 Catalyst Classified: Horizon={horizon} | Playbook={playbook} | {summary}")

                now_iso = datetime.now(UTC).isoformat()
                for db_file in ["trading_data.db", "trading_ledger.db"]:
                    try:
                        with sqlite3.connect(db_file, timeout=10.0) as conn:
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_catalyst', ?, ?)",
                                         (json.dumps(res), now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_playbook', ?, ?)",
                                         (playbook, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_horizon', ?, ?)",
                                         (horizon, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_category', ?, ?)",
                                         (category, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_headline', ?, ?)",
                                         (target_headline, now_iso))
                            conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_event_summary', ?, ?)",
                                         (summary, now_iso))

                            if playbook == "SECULAR_SHORT_BREAKDOWN":
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_bias', 'BEARISH', ?)",
                                             (now_iso,))
                                log.critical(f"[SENTINEL-EVENT] 🚨 Macro Bias immediately set to BEARISH due to {category} ({playbook})!")
                            elif playbook == "BULL_MOMENTUM_EXPANSION":
                                conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('ai_macro_bias', 'BULLISH', ?)",
                                             (now_iso,))
                                log.info(f"[SENTINEL-EVENT] 🚀 Macro Bias set to BULLISH due to {category} ({playbook})!")

                            conn.commit()
                    except Exception as pe:
                        log.warning(f"[SENTINEL-EVENT] Persistence error for {db_file}: {pe}")

                return res
        except Exception as e:
            log.error(f"[SENTINEL-EVENT] Catalyst evaluation error: {e}")

        return self._active_event_data

    # -----------------------------------------------------------------------
    # Hot Path & Async Directional Audit (with 2h max-age & WARNING size-down)
    # -----------------------------------------------------------------------
    def _check_playbook_override(self, direction: str) -> tuple[str, str, float] | None:
        """Evaluates 4h TTL active event catalyst playbooks."""
        dir_upper = direction.upper()
        if self._active_event_data:
            updated_at_str = self._active_event_data.get("updated_at")
            if updated_at_str:
                try:
                    updated_dt = datetime.fromisoformat(updated_at_str)
                    if updated_dt.tzinfo is None:
                        updated_dt = updated_dt.replace(tzinfo=UTC)
                    if (datetime.now(UTC) - updated_dt).total_seconds() > (4 * 3600):
                        self._active_event_playbook = "NORMAL_OPERATIONS"
                        self._active_event_summary = ""
                except Exception:
                    pass

        playbook = self._active_event_playbook
        if playbook == "SECULAR_SHORT_BREAKDOWN":
            if dir_upper == "LONG":
                return "VETO", f"[SECULAR BEAR VETO] Active structural bear catalyst: {self._active_event_summary}", 0.0
            elif dir_upper == "SHORT":
                return "CLEAR", f"[SECULAR SHORT CATALYST] Active structural bear catalyst validates short: {self._active_event_summary}", 1.0
        elif playbook == "FLASH_REBOUND_TRAP":
            if dir_upper == "SHORT":
                return "VETO", f"[FLASH TRAP VETO] Flash liquidity shock in progress (anti-bottom short shield): {self._active_event_summary}", 0.0
            elif dir_upper == "LONG":
                return "CLEAR", f"[REBOUND SNIPER CLEAR] Flash liquidation flush exhausting: {self._active_event_summary}", 1.0
        elif playbook == "BULL_MOMENTUM_EXPANSION":
            if dir_upper == "SHORT":
                return "VETO", f"[BULL CATALYST VETO] Major pro-crypto catalyst in progress: {self._active_event_summary}", 0.0
            elif dir_upper == "LONG":
                return "CLEAR", f"[BULL CATALYST CLEAR] Pro-crypto catalyst confirms expansion: {self._active_event_summary}", 1.0
        return None

    def _is_cache_entry_fresh(self, key: str) -> bool:
        entry = self._cache.get(key.upper())
        if not entry:
            return False
        _, _, fetched_at = entry
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        return (datetime.now(UTC) - fetched_at).total_seconds() <= MAX_CACHE_AGE_SECONDS

    async def async_perform_directional_audit(self, ticker: str, direction: str = "LONG") -> tuple[str, str, float]:
        """
        V43.0 Unified Async News & Sentiment Gate:
        Returns `(verdict, reason, size_multiplier)` where:
          - `verdict` in `("VETO", "WARNING", "CLEAR")`
          - `size_multiplier` is `0.0` (VETO/BLOCK), `0.5` (WARNING), or `1.0` (CLEAR/SAFE)
        Enforces a strict 2-hour freshness window (MAX_CACHE_AGE_SECONDS = 7200s).
        If memory cache and `sentiment_state.json` are both missing or >2h stale, performs an
        immediate live RSS refresh; if that also fails, fails CLOSED (`VETO`).
        """
        ticker_up = ticker.upper()
        if not (self._is_cache_entry_fresh("GLOBAL") and self._is_cache_entry_fresh(ticker_up)):
            loaded = await asyncio.to_thread(self._load_shared_sentiment_state)
            if not loaded or not (self._is_cache_entry_fresh("GLOBAL") and self._is_cache_entry_fresh(ticker_up)):
                log.info(f"[SENTINEL] Cache miss or >2h stale for {ticker_up} — triggering live RSS refresh...")
                await self.refresh_once([ticker_up])

        if not (self._is_cache_entry_fresh("GLOBAL") and self._is_cache_entry_fresh(ticker_up)):
            reason = f"[SENTINEL FAIL-CLOSED] Sentiment & RSS state missing or stale (>2h) for {ticker_up}."
            log.error(reason)
            return "VETO", reason, 0.0

        override = self._check_playbook_override(direction)
        if override is not None:
            return override

        g_status, g_msg, _ = self._cache["GLOBAL"]
        if g_status == "VETO":
            if direction.upper() == "SHORT" and self._active_event_playbook == "SECULAR_SHORT_BREAKDOWN":
                return "CLEAR", f"[GLOBAL EVENT CONFIRMED SHORT] {self._active_event_summary}", 1.0
            return "VETO", f"[GLOBAL SYSTEMIC VETO] {g_msg}", 0.0

        t_status, t_msg, fetched_at = self._cache[ticker_up]
        age_s = (datetime.now(UTC) - fetched_at).total_seconds()
        log.debug(f"[SENTINEL] Audit {ticker_up} (age={age_s:.0f}s): {t_status} — {t_msg}")

        if t_status == "VETO":
            return "VETO", t_msg, 0.0
        if g_status == "WARNING" or t_status == "WARNING":
            warn_msg = t_msg if t_status == "WARNING" else g_msg
            return "WARNING", warn_msg, 0.5
        return "CLEAR", t_msg, 1.0

    def perform_directional_audit(self, ticker: str, direction: str = "LONG") -> tuple[str, str]:
        """
        Synchronous directional audit wrapper returning `(verdict, reason)`.
        Enforces 2h max-age and fails closed if state is missing or stale.
        """
        override = self._check_playbook_override(direction)
        if override is not None:
            return override[0], override[1]
        return self.perform_disaster_audit(ticker, direction=direction)

    def perform_disaster_audit(self, ticker: str, direction: str = "LONG") -> tuple[str, str]:
        """
        V43.0 Synchronous lookup from memory cache or `sentiment_state.json`.
        Enforces 2-hour max-age (MAX_CACHE_AGE_SECONDS = 7200s) and fails CLOSED (`VETO`)
        if state is missing or stale.
        """
        ticker_up = ticker.upper()
        if not (self._is_cache_entry_fresh("GLOBAL") and self._is_cache_entry_fresh(ticker_up)):
            self._load_shared_sentiment_state()

        if not (self._is_cache_entry_fresh("GLOBAL") and self._is_cache_entry_fresh(ticker_up)):
            reason = f"[SENTINEL FAIL-CLOSED] Sentiment state missing or older than 2h for {ticker_up}."
            log.warning(reason)
            return "VETO", reason

        g_status, g_msg, _ = self._cache["GLOBAL"]
        if g_status == "VETO":
            if direction.upper() == "SHORT" and self._active_event_playbook == "SECULAR_SHORT_BREAKDOWN":
                return "CLEAR", f"[GLOBAL EVENT CONFIRMED SHORT] {self._active_event_summary}"
            log.critical(f"[SENTINEL-GLOBAL-VETO] Systemic disaster veto active: {g_msg}")
            return "VETO", f"[GLOBAL SYSTEMIC VETO] {g_msg}"

        status, msg, fetched_at = self._cache[ticker_up]
        age_s = (datetime.now(UTC) - fetched_at).total_seconds()
        log.debug(f"[SENTINEL] Cache hit for {ticker_up} (age={age_s:.0f}s): {status}")
        return status, msg

    # -----------------------------------------------------------------------
    # Async Fetch (used by the background loop, not the critical path)
    # -----------------------------------------------------------------------
    async def _async_disaster_audit(self, ticker: str) -> tuple[str, str]:
        """Non-blocking RSS fetch and keyword scan."""
        await self.refresh_once([ticker.upper()])
        return self.perform_disaster_audit(ticker)

    def _audit_global_headlines(self, headlines: list[str]) -> tuple[str, str]:
        """Scans all headlines for systemic market-wide disaster keywords using word boundaries."""
        import re
        for headline in headlines:
            headline_lower = headline.lower()
            if any(m in headline_lower for m in self.retrospective_markers):
                continue
            for systemic_kw in self.systemic_keywords:
                if re.search(rf"\b{re.escape(systemic_kw)}\b", headline_lower):
                    reason = f"SYSTEMIC MARKET DISASTER HEADLINE: '{headline}'"
                    log.critical(f"[SENTINEL] Global Systemic VETO triggered — {reason}")
                    return "VETO", reason
        return "CLEAR", f"Global macro news structurally safe ({len(headlines)} headlines scanned)."

    def _audit_ticker_headlines(self, ticker: str, headlines: list[str]) -> tuple[str, str]:
        """Scans global headlines for ticker-specific disaster (VETO) or caution (WARNING) keywords with word boundaries."""
        import re
        ticker_up = ticker.upper()
        # Ambiguous English words (LINK, NEAR, ADA, APT, SUI) must match uppercase ticker or full protocol name
        ambiguous_tickers = {"LINK", "NEAR", "ADA", "APT", "SUI"}
        expansions = {
            "BTC": ["bitcoin"],
            "SOL": ["solana"],
            "ETH": ["ethereum", "ether"],
            "XRP": ["ripple", "xrp ledger"],
            "DOGE": ["dogecoin"],
            "ADA": ["cardano"],
            "AVAX": ["avalanche"],
            "LINK": ["chainlink"],
            "BNB": ["binance", "bnb chain"],
            "SUI": ["sui network", "sui blockchain"],
            "NEAR": ["near protocol"],
            "APT": ["aptos"],
        }
        proto_names = expansions.get(ticker_up, [])

        matched_headlines = []
        for headline in headlines:
            h_lower = headline.lower()
            if any(m in h_lower for m in self.retrospective_markers):
                continue
            matched = False
            if ticker_up in ambiguous_tickers:
                # Require uppercase ticker token (e.g. LINK, $LINK) or full protocol name
                if re.search(rf"(?:\$|\b){re.escape(ticker_up)}\b", headline):
                    matched = True
            else:
                if re.search(rf"\b{re.escape(ticker_up.lower())}\b", h_lower):
                    matched = True
            if not matched and any(re.search(rf"\b{re.escape(p)}\b", h_lower) for p in proto_names):
                matched = True
            if matched:
                matched_headlines.append(headline)

        if not matched_headlines:
            return "CLEAR", f"No recent adverse news detected for {ticker_up} ({len(headlines)} global headlines scanned)."

        # 1. Check severe catastrophe keywords -> VETO (BLOCK)
        for headline in matched_headlines:
            headline_lower = headline.lower()
            for disaster in self.disaster_keywords:
                if re.search(rf"\b{re.escape(disaster)}\b", headline_lower):
                    reason = f"CATASTROPHIC HEADLINE DETECTED ({disaster}): '{headline}'"
                    log.warning(f"[SENTINEL] VETO cached for {ticker_up} — {reason}")
                    return "VETO", reason

        # 2. Check moderate caution keywords -> WARNING (0.5x size-down)
        for headline in matched_headlines:
            headline_lower = headline.lower()
            for warn_kw in self.warning_keywords:
                if re.search(rf"\b{re.escape(warn_kw)}\b", headline_lower):
                    reason = f"CAUTION HEADLINE DETECTED ({warn_kw}): '{headline}'"
                    log.info(f"[SENTINEL] WARNING (0.5x size-down) cached for {ticker_up} — {reason}")
                    return "WARNING", reason

        return "CLEAR", f"News for {ticker_up} structurally safe ({len(matched_headlines)} ticker headlines scanned)."

    async def _fetch_headlines_async(self, ticker: str = "") -> list[str]:
        """
        Fetches latest headlines from all RSS feeds.
        If a ticker is provided, prioritizes matching headlines; otherwise returns the full global set.
        """
        all_headlines = await self._fetch_all_headlines_global()
        if not ticker:
            return all_headlines

        target_keywords = [ticker.upper(), ticker.lower()]
        expansions = {"BTC": "bitcoin", "SOL": "solana", "ETH": "ethereum", "XRP": "ripple", "DOGE": "dogecoin"}
        if ticker.upper() in expansions:
            target_keywords.append(expansions[ticker.upper()])

        matched = [h for h in all_headlines if any(kw.lower() in h.lower() for kw in target_keywords)]
        return matched if matched else all_headlines

    def fetch_latest_headlines(self, ticker: str = "") -> list[str]:
        """Synchronous wrapper for retrieving headlines cleanly from external callers or Streamlit."""
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._fetch_headlines_async(ticker)).result(timeout=RSS_FETCH_TIMEOUT)
        except Exception as e:
            log.warning(f"[SENTINEL] Sync fetch error: {e}")
            return []

    async def _fetch_all_headlines_global(self) -> list[str]:
        """Fetches all RSS feeds once globally."""
        connector = aiohttp.TCPConnector(family=socket.AF_INET, resolver=aiohttp.ThreadedResolver())
        timeout = aiohttp.ClientTimeout(total=RSS_FETCH_TIMEOUT)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        headlines = []
        try:
            async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
                tasks = [self._fetch_one_feed_global(session, url) for url in self.news_feeds]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, list):
                        headlines.extend(result)
        except Exception as e:
            log.warning(f"[SENTINEL] global aiohttp session error: {e}")
        return list(dict.fromkeys(headlines))  # Preserves order while de-duplicating

    async def _fetch_one_feed_global(self, session: aiohttp.ClientSession, url: str) -> list[str]:
        """Fetches and parses one RSS feed XML, returning all titles."""
        import xml.etree.ElementTree as ET
        titles = []
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text(encoding="utf-8", errors="replace")
                root = ET.fromstring(text)
                for item in root.iter("item"):
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        clean_title = title_elem.text.strip()
                        if clean_title and clean_title not in titles:
                            titles.append(clean_title)
                if not titles:
                    titles = [el.text.strip() for el in root.iter("title") if el.text and el.text.strip()][:30]
        except Exception as e:
            log.debug(f"[SENTINEL] Global feed parse failed ({url}): {e}")
        return titles


# Backwards-compatibility aliases for caller components
SentinelAgent = SentimentSentinel
SentimentSentinel.evaluate_ticker_news = SentimentSentinel.perform_disaster_audit

# ---------------------------------------------------------------------------
# Standalone diagnostic
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC"

    async def _main():
        sentinel = SentimentSentinel()
        print(f"\n[SENTINEL] Running unified live refresh + directional audit for {ticker}...")
        await sentinel.refresh_once([ticker])
        verdict, msg, mult = await sentinel.async_perform_directional_audit(ticker, "LONG")
        print(f"\n{'='*60}")
        print(f"  Verdict         : {verdict}")
        print(f"  Size Multiplier : {mult}x")
        print(f"  Message         : {msg}")
        print(f"{'='*60}\n")

    asyncio.run(_main())
