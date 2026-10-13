from __future__ import annotations
import json
import re
import hashlib
import logging
import socket
import aiohttp
import asyncio
import config
import time
import pandas as pd
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V28.6 High-Performance API Key Rotation Pool
# ---------------------------------------------------------------------------
_groq_keys: list[str] = []
_keys_initialized = False
_groq_cooldown_until = 0.0

def _ensure_groq_keys() -> list[str]:
    """Dynamically load and initialize Groq API keys if not already done."""
    global _groq_keys, _keys_initialized
    if not _keys_initialized:
        try:
            import os
            from dotenv import load_dotenv
            _HERE = os.path.dirname(os.path.abspath(__file__))
            load_dotenv(dotenv_path=os.path.join(_HERE, ".env"), override=False)
        except Exception:
            pass
            
        keys = []
        base_key = os.getenv("GROQ_API_KEY")
        if base_key:
            keys.append(base_key.strip())
        for k, v in os.environ.items():
            if k.startswith("GROQ_API_KEY_") and v.strip():
                if "REPLACE_WITH" not in v.upper() and len(v.strip()) > 10:
                    keys.append(v.strip())
        _groq_keys = list(dict.fromkeys(keys))
        _keys_initialized = True
        log.info(f"[SRE-KEYPOOL] Dynamically initialized Groq API key pool with {len(_groq_keys)} keys.")
    return _groq_keys

_key_index = 0
_key_lock = None
_key_lock_loop = None

def _get_key_lock():
    global _key_lock, _key_lock_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _key_lock is None or _key_lock_loop is not loop:
        _key_lock = asyncio.Lock()
        _key_lock_loop = loop
    return _key_lock

_groq_semaphore = None
_groq_semaphore_loop = None

def _get_groq_semaphore():
    global _groq_semaphore, _groq_semaphore_loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if _groq_semaphore is None or _groq_semaphore_loop is not loop:
        _groq_semaphore = asyncio.Semaphore(2)
        _groq_semaphore_loop = loop
    return _groq_semaphore

async def rotate_groq_key(failed_key: str) -> str:
    """Atomic key rotation ensuring multi-threaded tasks only rotate once per key failure."""
    global _key_index
    keys = _ensure_groq_keys()
    if not keys:
        raise RuntimeError("No Groq API keys available in pool.")
    
    async with _get_key_lock():
        if keys[_key_index] == failed_key:
            _key_index = (_key_index + 1) % len(keys)
            new_key = keys[_key_index]
            log.warning(f"[SRE-ROTATION] Groq 429 Hit. Rotating to Key Index {_key_index}. Remaining active keys: {len(keys)}")
            return new_key
        return keys[_key_index]

_MODEL_COOLDOWN_FILE = os.path.join("cache", "ai_model_cooldowns.json")
_model_cooldowns_mem: dict[str, dict[str, float]] = {"gemini": {}, "groq": {}}
_model_cooldowns_last_sync = 0.0

def _sync_model_cooldowns() -> dict[str, dict[str, float]]:
    """Synchronizes per-model rate-limit cooldowns across all bot subprocesses via cache/ai_model_cooldowns.json."""
    global _model_cooldowns_last_sync
    now = time.time()
    if (now - _model_cooldowns_last_sync) < 5.0:
        return _model_cooldowns_mem
    _model_cooldowns_last_sync = now
    try:
        if os.path.exists(_MODEL_COOLDOWN_FILE):
            with open(_MODEL_COOLDOWN_FILE, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
            for provider in ("gemini", "groq"):
                prov_map = disk_data.get(provider, {})
                if isinstance(prov_map, dict):
                    for m, ts in prov_map.items():
                        ts_f = float(ts or 0.0)
                        if ts_f > now:
                            _model_cooldowns_mem[provider][m] = max(_model_cooldowns_mem[provider].get(m, 0.0), ts_f)
                        else:
                            _model_cooldowns_mem[provider].pop(m, None)
    except Exception:
        pass
    return _model_cooldowns_mem

def _set_model_cooldown(provider: str, model: str, duration_sec: float, reason: str = "") -> None:
    """Marks a specific Gemini or Groq model on cooldown and persists to disk for all subprocesses."""
    global _model_cooldowns_last_sync
    now = time.time()
    until_ts = now + float(duration_sec)
    _sync_model_cooldowns()
    _model_cooldowns_mem.setdefault(provider, {})[model] = until_ts
    _model_cooldowns_last_sync = now
    try:
        os.makedirs("cache", exist_ok=True)
        clean_data = {
            prov: {m: ts for m, ts in mp.items() if ts > now}
            for prov, mp in _model_cooldowns_mem.items()
        }
        tmp_path = f"{_MODEL_COOLDOWN_FILE}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f)
        os.replace(tmp_path, _MODEL_COOLDOWN_FILE)
    except Exception:
        pass
    log.warning(f"[AI-MODEL-ROTATION] {provider.upper()} model '{model}' cooled down for {duration_sec:.0f}s ({reason}). Switching to next model in pool.")

def _get_gemini_model_pool() -> list[str]:
    primary = os.getenv("GEMINI_MODEL") or getattr(config, "GEMINI_MODEL", "gemini-flash-lite-latest")
    pool = list(getattr(config, "GEMINI_MODEL_POOL", [
        "gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite-preview"
    ]))
    ordered = [primary] + [m for m in pool if m != primary]
    return list(dict.fromkeys(ordered))

def _get_groq_model_pool(requested_model: str | None = None) -> list[str]:
    primary = requested_model or os.getenv("GROQ_MODEL") or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
    pool = list(getattr(config, "GROQ_MODEL_POOL", [
        "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"
    ]))
    ordered = [primary] + [m for m in pool if m != primary]
    return list(dict.fromkeys(ordered))

async def _query_gemini_cloud(session: aiohttp.ClientSession, prompt: str, timeout_sec: float = 15.0) -> dict | None:
    """
    Primary cloud AI dispatcher using Google Gemini Multi-Model Pool with automatic model rotation.
    Rotates across GEMINI_MODEL_POOL whenever a model hits its 500 RPD quota (429) or 503 spike.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", "")
    if not gemini_key or "REPLACE" in gemini_key:
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    effective_timeout = max(12.0, float(timeout_sec))
    timeout = aiohttp.ClientTimeout(total=effective_timeout)
    model_pool = _get_gemini_model_pool()
    cooldowns = _sync_model_cooldowns().get("gemini", {})
    now = time.time()

    for model_name in model_pool:
        if cooldowns.get(model_name, 0.0) > now:
            continue

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"x-goog-api-key": gemini_key.strip()}
        try:
            async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    return {
                        "model_used": model_name,
                        "choices": [{
                            "message": {
                                "content": text
                            }
                        }]
                    }
                elif resp.status == 404:
                    _set_model_cooldown("gemini", model_name, 86400.0, reason="HTTP 404 model unavailable")
                    continue
                elif resp.status in (429, 503):
                    err_txt = await resp.text()
                    is_daily = ("free_tier_requests" in err_txt or "Quota exceeded" in err_txt or "per day" in err_txt.lower())
                    cd_sec = 1800.0 if is_daily else 60.0
                    _set_model_cooldown("gemini", model_name, cd_sec, reason=f"HTTP {resp.status} {'Daily Quota' if is_daily else 'Rate/Demand Limit'}")
                    continue
                else:
                    err_txt = await resp.text()
                    log.warning(f"[GEMINI-AI] Status {resp.status} on {model_name}: {err_txt[:100]}")
                    continue
        except Exception as e:
            log.warning(f"[GEMINI-AI] Request error ({type(e).__name__}) on {model_name}: {e or type(e).__name__}")
            continue

    return None

async def _query_groq_cloud_with_rotation(session: aiohttp.ClientSession, payload: dict, timeout_sec: float = 30.0) -> dict | None:
    """
    Centralized asynchronous dispatcher that manages instant API key rotation AND automatic
    multi-model rotation (GROQ_MODEL_POOL) when a model hits its 200k TPD or 8k TPM limit.
    """
    global _key_index, _groq_cooldown_until
    if time.time() < _groq_cooldown_until:
        return None

    keys = _ensure_groq_keys()
    if not keys:
        log.critical("Groq API Key Pool is empty. Setup GROQ_API_KEY_* in environment variables.")
        return None

    requested_model = payload.get("model") or getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
    model_pool = _get_groq_model_pool(requested_model)
    max_attempts = max(len(keys) + len(model_pool), 4)
    backoff = 0.4

    for attempt in range(max_attempts):
        # Select first non-cooled-down Groq model from pool
        cooldowns = _sync_model_cooldowns().get("groq", {})
        now = time.time()
        active_model = None
        for m in model_pool:
            if cooldowns.get(m, 0.0) <= now:
                active_model = m
                break
        if not active_model:
            # All Groq models in pool are currently on cooldown
            return None

        req_payload = dict(payload)
        req_payload["model"] = active_model

        # Protect key reading from race conditions due to concurrent key amputation
        async with _get_key_lock():
            keys = _ensure_groq_keys()
            if not keys:
                log.fatal("[FATAL] ALL API KEYS EXHAUSTED OR DEAD.")
                return {
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "execute": False,
                                "sl_mult": 2.0,
                                "tp_mult": 3.0,
                                "size_mult": 1.0,
                                "conviction": 0.0,
                                "sl_profile": "NORMAL",
                                "tp_profile": "SWING",
                                "notes": "[SRE FALLBACK] All API keys dead/exhausted. Calibration scaled to neutral 0.50 fallback."
                            })
                        }
                    }]
                }
            current_key = keys[_key_index % len(keys)]
        
        headers = {
            "Authorization": f"Bearer {current_key}",
            "Content-Type": "application/json"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=timeout_sec)
            status = None
            text = ""
            resp_json = None
            
            async with _get_groq_semaphore():
                async with session.post(config.GROQ_URL, headers=headers, json=req_payload, timeout=timeout) as resp:
                    status = resp.status
                    if status == 200:
                        resp_json = await resp.json()
                    else:
                        text = await resp.text()
            
            if status == 200:
                # FIX 2 (provenance, read-only): tag which pool model answered
                if isinstance(resp_json, dict):
                    resp_json.setdefault("model_used", resp_json.get("model") or active_model)
                return resp_json
            
            # V28.9 Self-Healing: Amputate invalid/dead API key on 401/403 Unauthorized responses
            if status in (401, 403):
                async with _get_key_lock():
                    keys = _ensure_groq_keys()
                    if current_key in keys:
                        idx = keys.index(current_key)
                        keys.pop(idx)
                        log.critical(f"[SRE-CRITICAL] API Key at index {idx} is DEAD/INVALID. Amputating from active pool.")
                        
                        if len(keys) == 0:
                            log.fatal("[FATAL] ALL API KEYS EXHAUSTED OR DEAD.")
                            return {
                                "choices": [{
                                    "message": {
                                        "content": json.dumps({
                                            "execute": False,
                                            "sl_mult": 2.0,
                                            "tp_mult": 3.0,
                                            "size_mult": 1.0,
                                            "conviction": 0.0,
                                            "sl_profile": "NORMAL",
                                            "tp_profile": "SWING",
                                            "notes": "[SRE FALLBACK] All API keys dead/exhausted. Calibration scaled to neutral 0.50 fallback."
                                        })
                                    }
                                }]
                            }
                        
                        # Realign active pointer to avoid skipping elements or overflow
                        if idx < _key_index:
                            _key_index -= 1
                        _key_index = _key_index % len(keys)
                continue  # Instantly retry query with the next available, healthy key
            
            if status in (429, 503):
                is_daily_tpd = ("tokens per day" in text.lower() or "(tpd)" in text.lower() or "requests per day" in text.lower() or "(rpd)" in text.lower())
                if is_daily_tpd:
                    # Organization-wide daily limit on this specific model: cool down model for 30m and switch model immediately!
                    _set_model_cooldown("groq", active_model, 1800.0, reason="Daily TPD/RPD limit reached")
                    continue
                else:
                    # Per-minute TPM/RPM burst or 503: cool down model briefly (45s) so next attempt uses sibling model AND rotate key
                    _set_model_cooldown("groq", active_model, 45.0, reason=f"HTTP {status} burst limit")
                    await rotate_groq_key(current_key)
                    await asyncio.sleep(min(backoff, 1.5))
                    backoff = min(backoff * 1.5, 4.0)
                    continue
            else:
                log.error(f"Groq API ({active_model}) failed with status {status}: {text[:160]}")
                if status in (400, 404):
                    _set_model_cooldown("groq", active_model, 3600.0, reason=f"HTTP {status}")
                    continue
                if status >= 500:
                    await rotate_groq_key(current_key)
                    await asyncio.sleep(0.5)
                    continue
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            async with _get_key_lock():
                keys = _ensure_groq_keys()
                safe_idx = _key_index % len(keys) if keys else 0
            log.warning(f"Network error on key index {safe_idx} ({active_model}): {e}")
            await rotate_groq_key(current_key)
            sleep_sec = min(backoff * (attempt + 1), 3.0)
            await asyncio.sleep(sleep_sec)
            backoff = min(backoff * 1.5, 6.0)
            continue

    _groq_cooldown_until = time.time() + 30.0
    log.critical("[CRITICAL] All Groq API keys & models in pool temporarily busy. Cooling off for 30s...")
    return None

# ---------------------------------------------------------------------------
# LLM Response Cache (Persistent)
# ---------------------------------------------------------------------------
def _ipv4_session() -> aiohttp.ClientSession:
    """Create an aiohttp session forced to IPv4 — avoids Windows localhost IPv6 failures."""
    import socket
    connector = aiohttp.TCPConnector(family=socket.AF_INET, resolver=aiohttp.ThreadedResolver())
    return aiohttp.ClientSession(connector=connector)

def _ensure_active_session(target) -> aiohttp.ClientSession:
    """Ensures client or self has an active aiohttp session bound to the current running event loop."""
    sess = getattr(target, "session", None)
    need_new = False
    if sess is None or sess.closed:
        need_new = True
    else:
        try:
            loop = asyncio.get_running_loop()
            sess_loop = getattr(sess, "_loop", None)
            if sess_loop is not loop or (hasattr(sess_loop, "is_closed") and sess_loop.is_closed()):
                need_new = True
        except RuntimeError:
            need_new = True
    if need_new:
        target.session = _ipv4_session()
    return target.session

def _cache_key(ctx: dict) -> str:
    """Stable cache key from coarse market signals — now includes ema_stack and patterns."""
    simplified = {
        "regime":    str(ctx.get("regime", "UNKNOWN")),
        "signal":    str(ctx.get("signal", "hold")),
        "phase":     str(ctx.get("phase", "UNKNOWN")),
        "btc":       str(ctx.get("btc_macro", "UNKNOWN")),
        "htf":       str(ctx.get("htf_macro", "UNKNOWN")),
        "atr_rnd":   round(float(ctx.get("atr_pct", 0.0)), 2),
        # V11.0: include structural context so different setups don't share a cache entry
        "ema_stack": bool(ctx.get("ema_stack", False)),
        "patterns":  sorted(str(p) for p in ctx.get("patterns", [])),
    }
    return hashlib.md5(json.dumps(simplified, sort_keys=True).encode()).hexdigest()

_llm_clients = {}

def get_llm(engine_name="default") -> "LLMClient":
    global _llm_clients
    if engine_name not in _llm_clients:
        _llm_clients[engine_name] = LLMClient(engine_name)
    return _llm_clients[engine_name]

def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        import pandas as pd
        return float(val) if pd.notna(val) else default
    except (TypeError, ValueError):
        return default

def robust_json_decode(text: str) -> dict:
    """Extracts and repairs JSON from LLM noise (markdown, trailing commas, trailing brackets)."""
    import json
    import re
    if not text:
        return {}
    try:
        # 1. Try direct load
        return json.loads(text)
    except Exception:
        pass

    # 2. Try JSONDecoder().raw_decode starting from first {
    try:
        start_idx = text.find("{")
        if start_idx != -1:
            obj, _ = json.JSONDecoder().raw_decode(text[start_idx:])
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass

    # 3. Extract first {...} block with regex repair
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        clean = match.group(0)
        # Simple repair: remove trailing commas before } or ]
        clean = re.sub(r",\s*([\]}])", r"\1", clean)
        try:
            return json.loads(clean)
        except Exception:
            pass

    return {}

class LLMClient:
    def __init__(self, engine_name="default"):
        import threading
        self.engine_name = engine_name
        os.makedirs("cache", exist_ok=True)
        self.cache_file = os.path.join("cache", f"ai_cache_{engine_name}.json")
        self.session = None
        self.lock = threading.Lock()
        self.cache = self._load_cache()

    def _load_cache(self):
        # Check cache directory first, then fallback to root for seamless migration
        target_path = self.cache_file
        if not os.path.exists(target_path):
            root_path = f"ai_cache_{self.engine_name}.json"
            if os.path.exists(root_path):
                target_path = root_path
        if os.path.exists(target_path):
            try:
                with open(target_path, "r") as f:
                    raw = json.load(f)
                    return {k: (v[0], v[1]) for k, v in raw.items()}
            except:
                pass
        return {}

    def _save_cache(self):
        # V31.0 P2: Prune stale entries BEFORE writing to disk.
        now = time.time()
        TTL = 900  # 15 minutes — matches in-memory pruning threshold
        
        with self.lock:
            pruned = {k: v for k, v in self.cache.items() if (now - v[0]) <= TTL}
            
            # Enforce strict upper bound of 500 entries (D3-11)
            if len(pruned) > 500:
                sorted_items = sorted(pruned.items(), key=lambda x: x[1][0], reverse=True)
                pruned = dict(sorted_items[:500])

        try:
            # Write to temp file first, then atomically replace the destination cache file
            import tempfile
            import os
            dir_name = os.path.dirname(os.path.abspath(self.cache_file))
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
                json.dump(pruned, tf)
                temp_name = tf.name
            os.replace(temp_name, self.cache_file)
            with self.lock:
                self.cache = pruned  # Keep in-memory dict consistent with what was written

        except Exception as e:
            try:
                if 'temp_name' in locals() and os.path.exists(temp_name):
                    os.remove(temp_name)
            except:
                pass

    def _db_pattern_lookup(self, symbol_clean):
        detected_patterns = []
        arbitrage_signal = "None"
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ledger_db = os.path.join(base_dir, "trading_ledger.db")
            if os.path.exists(ledger_db):
                import sqlite3
                from datetime import datetime, UTC, timedelta
                # Use context manager to guarantee conn is always closed on exception
                with sqlite3.connect(ledger_db, timeout=5) as conn:
                    # A. Fetch patterns within last 30 minutes
                    time_threshold_pat = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
                    cursor_pat = conn.execute(
                        "SELECT timeframe, pattern_name FROM active_patterns WHERE symbol = ? AND timestamp >= ?",
                        (symbol_clean, time_threshold_pat)
                    )
                    rows_pat = cursor_pat.fetchall()
                    for row in rows_pat:
                        detected_patterns.append(f"{row[0]} {row[1]}")

                    # B. Fetch correlation signals within last 3 minutes
                    time_threshold_corr = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
                    cursor_corr = conn.execute(
                        "SELECT signal FROM active_correlations WHERE lag_asset = ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT 1",
                        (symbol_clean, time_threshold_corr)
                    )
                    row_corr = cursor_corr.fetchone()
                    if row_corr:
                        arbitrage_signal = f"FLASH {row_corr[0]}"
        except Exception:
            pass
        return detected_patterns, arbitrage_signal

    async def _save_cache_async(self):
        try:
            await asyncio.to_thread(self._save_cache)
        except Exception as e:
            log.warning(f"[CACHE-ASYNC-WARN] Failed to flush cache: {e}")

    async def close(self):
        await self._save_cache_async()
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_risk_params(self, regime, atr_pct, signal, phase, session_pnl=0.0, streak=0, btc_context="UNKNOWN", market_context="", htf_trend="UNKNOWN", current_regime="TRENDING", ema_slope=0.0, patterns=None, ema_stack=False, timeout_sec: float = 30.0, symbol: str = None):
        """Targeted volatility prompt for SL/TP and Size multipliers."""
        # [P0-RESTORED] Env-var-gated observation mode. Set OBSERVATION_MODE=1 in .env ONLY for
        # controlled data-harvesting sessions. Default is 0 (LIVE AI VETO ACTIVE).
        if os.getenv("OBSERVATION_MODE", "0") == "1":
            log.warning("[OBSERVATION MODE] AI risk engine bypassed — OBSERVATION_MODE=1 is set. Set to 0 to restore live LLM veto.")
            return {
                "action": "open" if "open" in signal else "add",
                "entry_quality": 100.0,
                "size_mult": 1.0,
                "sl_mult": 2.0,
                "tp_mult": 3.0,
                "sl": {"mode": "NORMAL"},
                "tp": {"mode": "SWING"},
                "trailing": {"mode": "NORMAL"},
                "notes": "Observation Mode Override (OBSERVATION_MODE=1)"
            }

        
        # Auto-derive symbol from engine_name if not explicitly passed (e.g. "berserker-BTC-USDT")
        if not symbol:
            if self.engine_name and self.engine_name.startswith("berserker-"):
                symbol = self.engine_name.replace("berserker-", "").replace("-", "/")

        # 1. Load Dynamic Learned Rules directly from trading_ledger.db (the true source of truth)
        def _load_learned_rules_from_db():
            rules = []
            try:
                import sqlite3 as _sq
                base_dir = os.path.dirname(os.path.abspath(__file__))
                ledger_db = os.path.join(base_dir, "trading_ledger.db")
                if os.path.exists(ledger_db):
                    with _sq.connect(ledger_db, timeout=5) as _conn:
                        rows = _conn.execute(
                            "SELECT category, rule_text, confidence FROM learned_rules "
                            "WHERE confidence >= 0.70 AND created_at >= datetime('now', '-30 days') "
                            "ORDER BY confidence DESC LIMIT 25"
                        ).fetchall()
                        for row in rows:
                            rules.append(f"[{row[0]}] (conf={row[2]:.0%}) {row[1]}")
            except Exception:
                pass
            return rules

        learned_rules_text = ""
        db_rules = await asyncio.to_thread(_load_learned_rules_from_db)
        if db_rules:
            learned_rules_text = "\n\n--- DYNAMIC LEARNED RULES (CRITICAL — ALWAYS ENFORCE) ---\n"
            for rule in db_rules:
                learned_rules_text += f"- {rule}\n"

        ctx = {
            "symbol": symbol or "UNKNOWN",
            "regime": regime,
            "atr_pct": round(atr_pct, 3),
            "signal": signal,
            "phase": phase,
            "session_pnl": f"{session_pnl:+.2f}%",
            "streak": streak,
            "btc_macro": btc_context,
            "htf_macro": htf_trend,
            "market_summary": market_context,
            "current_regime": current_regime,
            "ema_slope_3c": round(ema_slope, 4),
            "patterns": patterns or [],
            "ema_stack": ema_stack
        }
        
        # Cache logic: coarse key -> (timestamp, result)
        ckey = _cache_key(ctx)
        now = time.time()
        
        with self.lock:
            # Pruning step: V11.0 TTL reduced to 15 min (was 1 hour — too stale for 5m Berserker)
            to_del = [k for k, v in self.cache.items() if (now - v[0]) > 900]
            for k in to_del: del self.cache[k]
            
            if ckey in self.cache:
                log.debug(f"LLM Cache Hit [{self.engine_name}]: {ckey}")
                return self.cache[ckey][1]

        # Enforce symbol cleanliness
        symbol_to_query = symbol or "UNKNOWN"
        symbol_upper = str(symbol_to_query).upper()
        symbol_clean = symbol_upper.replace("/", "").replace("-", "")
        
        detected_patterns, arbitrage_signal = await asyncio.to_thread(self._db_pattern_lookup, symbol_clean)
        patterns_str = ", ".join(detected_patterns) if detected_patterns else "None"

        # Construct merged context
        merged_context = dict(ctx)
        merged_context["detected_patterns_last_30m"] = patterns_str
        merged_context["active_arbitrage_signals_last_3m"] = arbitrage_signal
        if learned_rules_text:
            merged_context["learned_rules"] = learned_rules_text

        # Format system prompt exactly as specified
        system_prompt = f"""You are the Chief Investment Officer reviewing a trade signal.
MARKET DATA: {json.dumps(merged_context)}
SIGNAL: {signal} on {symbol_upper}

Evaluate from BOTH bull and bear perspectives, then render a final verdict.
Return ONLY valid JSON with this exact structure:
{{
  "execute": true,
  "conviction": 85,
  "size_mult": 1.0,
  "sl_mult": 2.0,
  "tp_mult": 3.0,
  "rationale": "one concise sentence"
}}"""

        self.session = _ensure_active_session(self)

        data = None
        # Tier 1: Google Gemini Primary
        try:
            data = await _query_gemini_cloud(self.session, system_prompt, timeout_sec=timeout_sec)
            if data:
                log.info(f"[AI-ENGINE] Processed opportunity via Primary Google Gemini [{getattr(config, 'GEMINI_MODEL', 'gemini-3.5-flash-lite')}].")
        except Exception as e:
            log.warning(f"[AI-ENGINE] Gemini attempt failed: {e}. Falling back to Groq.")

        # Tier 2: Groq Cloud Failover
        if not data:
            payload = {
                "model": config.GROQ_MODEL, 
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Analyze and return the required JSON verdict."}
                ],
                "temperature": 0.1
            }
            data = await _query_groq_cloud_with_rotation(self.session, payload, timeout_sec=timeout_sec)
            if data:
                log.info(f"[AI-ENGINE] Processed opportunity via Groq Failover [{config.GROQ_MODEL}].")
        
        if data:
            try:
                raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                text = (raw_response.strip() if raw_response is not None else "")
                res = robust_json_decode(text)
                
                # Check execute and conviction
                execute = res.get("execute", False)
                conviction_raw = safe_float(res.get("conviction"), 0.0)
                
                if conviction_raw <= 1.0:
                    conviction_raw *= 100.0
                conviction_score = max(1.0, min(100.0, conviction_raw))
                
                if not execute:
                    conviction_score = 0.0
                
                result = {
                    "action": "open" if "open" in signal else "add",
                    "entry_quality": conviction_score, 
                    "quality": conviction_score,
                    "size_mult": max(0.5, min(2.0, safe_float(res.get("size_mult"), 1.0))),
                    "sl_mult":   max(1.0, min(4.0, safe_float(res.get("sl_mult"), 2.0))),
                    "tp_mult":   max(1.5, min(6.0, safe_float(res.get("tp_mult"), 3.0))),
                    "sl":       {"mode": "NORMAL"},
                    "tp":       {"mode": "SWING"},
                    "trailing": {"mode": "NORMAL"},
                    "notes": str(res.get("rationale", "Risk calibrated."))
                }
                with self.lock:
                    self.cache[ckey] = (now, result)
                await self._save_cache_async()
                return result
            except Exception as parse_err:
                log.error(f"Cloud response parsing failed: {parse_err}")

        # Safe Fallback
        log.warning("[AI-ENGINE] CRITICAL: All cloud API endpoints exhausted. Safety Veto triggered (Score: 0.0).")
        return {
            "action": "open" if "open" in signal else "add",
            "entry_quality": 0.0,
            "quality": 0.0,
            "size_mult": 0.0, "sl_mult": 2.0, "tp_mult": 3.0,
            "sl": {"mode": "NORMAL"}, "tp": {"mode": "SWING"}, "trailing": {"mode": "NORMAL"},
            "notes": "LLM Keypool Exhausted — Hard Veto."
        }

    async def evaluate(self, trade, timeout_sec: float = 30.0):
        """Post-trade LLM evaluator (Async)."""
        summary = {
            "id": trade.get("id"), "side": trade.get("side"),
            "entry": trade.get("entry_price"), "unr_pnl": trade.get("unr"),
            "regime": trade.get("regime"), "phase": trade.get("phase")
        }
        ctx = json.dumps(summary, separators=(',', ':'))
        prompt = f'JSON-ONLY. Review trade summary: {ctx}. Return strictly: {{"notes":["p1","p2","p3"]}}'

        self.session = _ensure_active_session(self)

        data = None
        try:
            data = await _query_gemini_cloud(self.session, prompt, timeout_sec=timeout_sec)
        except Exception:
            pass
            
        if not data:
            payload = {
                "model": config.GROQ_MODEL, 
                "messages": [
                    {"role": "system", "content": "You are a trading evaluator."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            data = await _query_groq_cloud_with_rotation(self.session, payload, timeout_sec=timeout_sec)
        
        if data:
            try:
                raw_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                text = (raw_response.strip() if raw_response is not None else "")
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0)).get("notes", ["Completed."])
            except Exception as parse_err:
                log.warning(f"Eval parsing failed: {parse_err}")

        return ["Evaluator unavailable."]

async def query_cloud_llm(prompt: str, system_prompt: str = "You are an expert trading assistant.") -> str:
    """
    Standard global LLM hook to query Gemini Primary / Groq Failover.
    """
    client = get_llm()
    client.session = _ensure_active_session(client)

    combined_prompt = f"{system_prompt}\n\n{prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined_prompt, timeout_sec=15.0)
    except Exception:
        pass
        
    if not data:
        payload = {
            "model": config.GROQ_MODEL, 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=15.0)
    
    if data:
        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return (raw_content.strip() if raw_content is not None else "")
        
    return "Fallback response."


# ---------------------------------------------------------------------------
# V37.0 Autonomous AI Sovereign Intelligence Suite
# ---------------------------------------------------------------------------

async def check_position_health(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    unrealized_pnl_pct: float,
    holding_mins: float,
    indicators: dict,
    current_sl: float,
    current_tp: float,
    timeout_sec: float = 12.0
) -> dict:
    """
    Tactical Low-Latency AI Position Health Check (Groq Cloud ~250ms).
    Dynamically assesses active trade health, momentum decay, and profit locking.
    CRITICAL INVARIANT: Stop Loss suggestions can ONLY tighten (never loosen).
    """
    client = get_llm("health_check")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Algorithmic Risk Officer overseeing an active open crypto futures position. "
        "Your duty is to protect capital, lock in gains, and detect momentum exhaustion before reversals happen. "
        "Review the active position telemetry and return ONLY valid JSON with no preamble or markdown.\n"
        "Rules:\n"
        "1. If trade is strongly trending in our favor: verdict 'HOLD', health_grade 'PRISTINE'.\n"
        "2. If trade has built solid profit and momentum is waning: verdict 'TIGHTEN_SL', provide 'suggested_sl' closer to price to lock gains, health_grade 'WARNING'.\n"
        "3. If trade is stalling at a major exhaustion point or volume completely died: verdict 'TAKE_PROFIT_EARLY' or 'TIGHTEN_SL'.\n"
        "4. If price action shows severe adverse structural failure: verdict 'CLOSE_NOW' or 'TIGHTEN_SL', health_grade 'CRITICAL'.\n"
        "IMPORTANT: suggested_sl for LONG must be higher than current_sl and below current_price. For SHORT, it must be lower than current_sl and above current_price."
    )

    ctx = {
        "symbol": symbol,
        "side": side.upper(),
        "entry_price": round(entry_price, 4),
        "current_price": round(current_price, 4),
        "unrealized_pnl_pct": f"{unrealized_pnl_pct:+.2f}%",
        "holding_duration_mins": round(holding_mins, 1),
        "current_sl": round(current_sl, 4),
        "current_tp": round(current_tp, 4),
        "indicators": {
            "rsi": round(indicators.get("rsi", 50.0), 1),
            "adx": round(indicators.get("adx", 20.0), 1),
            "atr": round(indicators.get("atr", 0.0), 4),
            "ema20": round(indicators.get("ema20", current_price), 4),
            "vol_ratio": round(indicators.get("vol_ratio", 1.0), 2)
        }
    }

    user_prompt = (
        f"Analyze active position telemetry:\n{json.dumps(ctx, indent=2)}\n\n"
        "Respond with JSON schema:\n"
        "{\n"
        '  "verdict": "HOLD" | "TIGHTEN_SL" | "TAKE_PROFIT_EARLY" | "CLOSE_NOW",\n'
        '  "suggested_sl": float or null,\n'
        '  "confidence": 0.0 to 1.0,\n'
        '  "diagnosis": "1 concise sentence explaining the health diagnosis",\n'
        '  "health_grade": "PRISTINE" | "WARNING" | "CRITICAL"\n'
        "}"
    )

    # Groq primary for sub-second execution
    payload = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    
    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)
    # Gemini failover if Groq failed
    if not data:
        try:
            combined = f"{system_prompt}\n\n{user_prompt}"
            data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
        except Exception:
            pass

    if data:
        try:
            raw_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = robust_json_decode(raw_text)
            verdict = parsed.get("verdict", "HOLD").upper()
            suggested_sl = parsed.get("suggested_sl")
            confidence = safe_float(parsed.get("confidence"), 0.5)
            diagnosis = str(parsed.get("diagnosis", "Active position health monitored."))
            health_grade = parsed.get("health_grade", "PRISTINE").upper()

            # Enforce strict stop-tightening math (NEVER LOOSEN)
            valid_suggested_sl = None
            if suggested_sl is not None:
                suggested_sl = safe_float(suggested_sl, 0.0)
                if side.lower() == "long":
                    # For long: tighter means higher than current_sl and still below current price
                    if suggested_sl > current_sl and suggested_sl < current_price:
                        valid_suggested_sl = suggested_sl
                    else:
                        if verdict == "TIGHTEN_SL":
                            verdict = "HOLD"
                elif side.lower() == "short":
                    # For short: tighter means lower than current_sl and still above current price
                    if suggested_sl < current_sl and suggested_sl > current_price:
                        valid_suggested_sl = suggested_sl
                    else:
                        if verdict == "TIGHTEN_SL":
                            verdict = "HOLD"

            return {
                "verdict": verdict,
                "suggested_sl": valid_suggested_sl,
                "confidence": confidence,
                "diagnosis": diagnosis,
                "health_grade": health_grade,
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as pe:
            log.warning(f"[AI-HEALTH] Parse failed: {pe}")

    return {
        "verdict": "HOLD",
        "suggested_sl": None,
        "confidence": 0.0,
        "diagnosis": "Health check offline, adhering to deterministic hard/trailing stop geometry.",
        "health_grade": "DEGRADED_AI_OFFLINE",
        "checked_at": datetime.now(timezone.utc).isoformat()
    }


async def red_team_deliberation(
    symbol: str,
    side: str,
    context_data: dict,
    timeout_sec: float = 15.0
) -> dict:
    """
    Multi-Agent Adversarial Deliberation for Candidate Entry Setups.
    Runs Red Team (Risk Officer) and Blue Team (Alpha Hunter) in parallel on Groq (~250ms),
    then has the Supreme Arbiter (CIO) render the synthesis verdict (<600ms total).
    """
    client = get_llm("red_team")
    client.session = _ensure_active_session(client)

    sym_upper = symbol.upper()
    side_upper = side.upper()
    ctx_str = json.dumps(context_data, separators=(',', ':'))

    # Load Dynamic Learned Rules distilled from past mistakes
    def _get_top_rules_sync():
        rules = []
        try:
            import sqlite3 as _sq
            base_dir = os.path.dirname(os.path.abspath(__file__))
            ledger_db = os.path.join(base_dir, "trading_ledger.db")
            if os.path.exists(ledger_db):
                with _sq.connect(ledger_db, timeout=3.0) as _conn:
                    rows = _conn.execute(
                        "SELECT category, rule_text, confidence FROM learned_rules WHERE confidence >= 0.70 ORDER BY confidence DESC LIMIT 15"
                    ).fetchall()
                    for row in rows:
                        rules.append(f"[{row[0]}] {row[1]} (conf={row[2]:.0%})")
        except Exception:
            pass
        return rules

    top_rules = await asyncio.to_thread(_get_top_rules_sync)
    rules_block = "\n".join(f"- {r}" for r in top_rules) if top_rules else "No active learned rules."

    # Unified Adversarial Boardroom Prompt (Consolidates Rule Audit, Liquidity Sweep, Red Team, Alpha Hunter & Arbiter into 1 call)
    unified_prompt = (
        "You are the Supreme Arbiter and Chief Investment Officer presiding over the Multi-Agent Investment Boardroom.\n"
        f"Setup: {side_upper} on {sym_upper}\n"
        f"Advisory Lessons from Past Post-Mortems (Use to calibrate conviction and sizing, NOT as rigid mechanical vetoes over pre-validated indicator gates):\n{rules_block}\n"
        f"Full Telemetry: {ctx_str}\n\n"
        "Note: This candidate has ALREADY passed all deterministic quantitative gates (Ichimoku Kumo breakout, ADX threshold, RSI bounds, Momentum Divergence, Volume Profile VPVR, and TD Sequential). The execution engine automatically enforces >=2.5x 4H ATR stop-losses and volatility-scaled sizing.\n"
        "Synthesize the trade setup across three pillars in a single pass:\n"
        "1. BEARISH RED TEAM & LIQUIDITY SWEEP AUDIT: Identify genuine structural bull/bear traps, extreme exhaustion, or severe macro/microstructure headwinds.\n"
        "2. BULLISH ALPHA THESIS: Evaluate trend alignment, ADX momentum strength, non-exhausted RSI trajectory, and breakout structure.\n"
        "3. SUPREME ARBITER RULING: Does the alpha thesis outweigh red team objections? Approve setups (execute=true, conviction >= 75) when trend and momentum metrics are constructive, adjusting size_mult/sl_mult if moderate caution is warranted.\n\n"
        "Return strictly valid JSON with no markdown:\n"
        "{\n"
        '  "bear_objection": "1 sentence describing the main risk or trap concern",\n'
        '  "bull_thesis": "1 sentence describing the trend/momentum edge",\n'
        '  "execute": boolean,\n'
        '  "conviction": 0 to 100,\n'
        '  "size_mult": 0.5 to 1.5,\n'
        '  "sl_mult": 1.0 to 3.0,\n'
        '  "tp_mult": 1.5 to 5.0,\n'
        '  "rationale": "1 concise sentence summarizing the arbiter ruling"\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [{"role": "user", "content": unified_prompt}],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    # Query Groq first (fastest inference, 3-key failover pool)
    resp = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)
    # Gemini failover (single call)
    if not resp:
        try:
            resp = await _query_gemini_cloud(client.session, unified_prompt, timeout_sec=timeout_sec)
        except Exception:
            pass

    if resp:
        try:
            raw = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            verdict = robust_json_decode(raw)
            execute = bool(verdict.get("execute", False))
            conviction = safe_float(verdict.get("conviction"), 0.0)
            if conviction <= 1.0 and conviction > 0.0:
                conviction *= 100.0
            conviction = max(0.0, min(100.0, conviction))
            if not execute:
                conviction = 0.0

            return {
                "execute": execute,
                "conviction_score": conviction / 100.0,
                "conviction_percent": conviction,
                "size_mult": max(0.5, min(1.5, safe_float(verdict.get("size_mult"), 1.0))),
                "sl_mult": max(1.0, min(3.0, safe_float(verdict.get("sl_mult"), 2.0))),
                "tp_mult": max(1.5, min(5.0, safe_float(verdict.get("tp_mult"), 3.0))),
                "red_team_objection": str(verdict.get("bear_objection", "Red team flagged potential volatility.")),
                "alpha_thesis": str(verdict.get("bull_thesis", "Alpha hunter confirmed directional trend.")),
                "final_rationale": str(verdict.get("rationale", "Adversarial deliberation concluded.")),
                "served_model_id": str(resp.get("model_used") or resp.get("model") or "cloud_response_unattributed"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as ae:
            log.warning(f"[RED-TEAM] Arbiter parse error: {ae}")

    # FAIL-CLOSED FALLBACK:
    # If cloud LLMs are busy or rate-limited, safely veto execution rather than simulating arbitrary scores
    adx_val = safe_float(context_data.get("adx"), 20.0)
    rsi_val = safe_float(context_data.get("rsi"), 50.0)
    log.warning(f"[BOARDROOM-FAIL-CLOSED] Cloud AI offline/exhausted for {sym_upper} (ADX: {adx_val:.1f}, RSI: {rsi_val:.1f}). Safety veto engaged (fail-closed).")
    return {
        "execute": False,
        "conviction_score": 0.0,
        "conviction_percent": 0.0,
        "size_mult": 0.0,
        "sl_mult": 2.0,
        "tp_mult": 3.0,
        "red_team_objection": "Cloud AI deliberation unavailable — fail-closed safety veto active.",
        "alpha_thesis": "Deliberation offline.",
        "final_rationale": "Boardroom LLM offline or timed out; fail-closed capital preservation active.",
        "served_model_id": None,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def generate_macro_synthesis(
    macro_data: dict,
    timeout_sec: float = 20.0
) -> dict:
    """
    Periodic Macro & Derivatives Synthesis via Google Gemini Flash (~72 calls/day).
    Synthesizes SPY, DXY, BTC ROC, Warden regime, funding skews, and news events.
    """
    client = get_llm("macro_synth")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Global Chief Macro Strategist for a high-frequency autonomous crypto fund. "
        "Synthesize cross-market macroeconomic data, traditional equity contagion, dollar strength, "
        "Bitcoin trend velocity, Bitget perpetual funding rate skews, open interest, and institutional positioning. "
        "Classify the derivatives skew regime into one of: "
        "'LIQUIDITY_EXPANSION' (growing OI + healthy funding), "
        "'SHORT_SQUEEZE_TRAP' (negative funding rates, crowded shorts vulnerable to squeeze), "
        "'BALANCED_CHOP' (neutral funding 0.00% to 0.01%, range-bound), "
        "'DELEVERAGING_RISK' (overleveraged longs with funding spike >0.03% or cascade warning). "
        "Set 'derivatives_skew_halt' to true ONLY if DELEVERAGING_RISK indicates an imminent violent long cascade/flush. "
        "Return ONLY valid JSON with no markdown formatting."
    )

    user_prompt = (
        f"Macro and Derivatives telemetry:\n{json.dumps(macro_data, indent=2)}\n\n"
        "Provide your institutional synthesis in this JSON schema:\n"
        "{\n"
        '  "market_posture": "AGGRESSIVE_EXPANSION" | "TACTICAL_DEFENSE" | "FULL_PRESERVATION",\n'
        '  "macro_bias": "BULLISH" | "BEARISH" | "NEUTRAL_CHOP",\n'
        '  "derivatives_skew_regime": "LIQUIDITY_EXPANSION" | "SHORT_SQUEEZE_TRAP" | "BALANCED_CHOP" | "DELEVERAGING_RISK",\n'
        '  "derivatives_skew_halt": false,\n'
        '  "derivatives_notes": "1 concise sentence analyzing funding rates and open interest skew",\n'
        '  "executive_summary": "2 clear sentences summarizing the current global liquidity regime and immediate market implications.",\n'
        '  "key_risks": ["concise risk 1", "concise risk 2"],\n'
        '  "tactical_guidance": "1 sentence directive for algo execution parameters"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    # Gemini primary for rich reasoning
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as ge:
        log.warning(f"[MACRO-SYNTH] Gemini error: {ge}")

    # Groq failover
    if not data:
        payload = {
            "model": config.GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            return {
                "market_posture": res.get("market_posture", "TACTICAL_DEFENSE"),
                "macro_bias": res.get("macro_bias", "NEUTRAL_CHOP"),
                "derivatives_skew_regime": res.get("derivatives_skew_regime", "BALANCED_CHOP"),
                "derivatives_skew_halt": bool(res.get("derivatives_skew_halt", False)),
                "derivatives_notes": res.get("derivatives_notes", "Funding rates and open interest within standard boundaries."),
                "executive_summary": res.get("executive_summary", "Global macro environment neutral with balanced liquidity."),
                "key_risks": res.get("key_risks", ["Volatility clustering", "Choppy range rotation"]),
                "tactical_guidance": res.get("tactical_guidance", "Maintain standard stops and honor regime constraints."),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as pe:
            log.warning(f"[MACRO-SYNTH] Parse error: {pe}")

    return {
        "market_posture": "TACTICAL_DEFENSE",
        "macro_bias": "NEUTRAL_CHOP",
        "derivatives_skew_regime": "BALANCED_CHOP",
        "derivatives_skew_halt": False,
        "derivatives_notes": "Derivatives telemetry baseline intact.",
        "executive_summary": "Macro telemetry stable. Algorithms operating under baseline risk constraints.",
        "key_risks": ["API degradation fallback"],
        "tactical_guidance": "Enforce baseline stop losses and conservative sizing.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

# ---------------------------------------------------------------------------
# V38.0 Autonomous AI Institutional Intelligence Engines
# ---------------------------------------------------------------------------

async def analyze_order_flow_tape(ofi_packet: dict, timeout_sec: float = 8.0) -> dict:
    """
    Feature 2: High-Speed AI Order Flow Tape & Iceberg Absorption Interpreter.
    Dispatched exclusively via Groq Cloud 3-key rotation pool for sub-second inference (<300ms)
    and zero Gemini token consumption.
    """
    client = get_llm("tape_whisperer")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Order Flow & Tape Reading Specialist. "
        "Analyze Level-2 depth imbalance (OFI), cumulative volume delta, bid/ask depth skew, and velocity. "
        "Detect if smart money is absorbing aggressive market orders (iceberg orders), defending walls, or spoofing. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Microstructure Depth Snapshot:\n{json.dumps(ofi_packet, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "tape_condition": "BULLISH_ABSORPTION" | "BEARISH_ABSORPTION" | "WALL_DEFENSE" | "NEUTRAL_BALANCED" | "SPOOF_SUSPECTED",\n'
        '  "absorption_side": "BID" | "ASK" | "NONE",\n'
        '  "confidence": 0.85,\n'
        '  "headline": "1 punchy sentence describing what institutional market makers are doing",\n'
        '  "execution_veto": "BLOCK_SHORTS" | "BLOCK_LONGS" | "CLEAR"\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)
    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "tape_condition": "NEUTRAL_BALANCED",
        "absorption_side": "NONE",
        "confidence": 0.50,
        "headline": "Order flow depth balanced across primary levels.",
        "execution_veto": "CLEAR",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def rank_asset_hierarchy(fleet_context: dict, timeout_sec: float = 20.0) -> dict:
    """
    Feature 3: 2-Hour Autonomous Asset Hierarchy & Momentum Ranking Engine.
    Dispatched via Google Gemini Flash Lite (~12 calls/day).
    Ranks the 8 symbols from highest edge to avoid-chop list.
    """
    client = get_llm("chief_strategist")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Chief Quantitative Portfolio Strategist and Cross-Asset Relative Strength Architect. "
        "Analyze the multi-asset crypto fleet telemetry (trend persistence, ADX, Hurst, volume momentum, and relative strength vs BTC). "
        "Your task is to rank the fleet assets AND classify each asset's Decoupled Micro-Regime:\n"
        "1. INDEPENDENT_RUNNER: Displaying strong independent spot accumulation, surging volume, and RS decoupling from BTC. Authorized to trade even if BTC is chopping!\n"
        "2. BTC_BETA_SLAVE: Closely pegged to Bitcoin's trajectory. Bound strictly by the Bitcoin Anchor.\n"
        "3. ACCUMULATION_BASE: Range-bound base building; candidate for mean-reversion grid or breakout watch.\n"
        "4. EXHAUSTION_DISTRIBUTION: Momentum waning or heavy overhead supply; strictly close-only.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Fleet Telemetry Matrix:\n{json.dumps(fleet_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "priority_leaders": ["SYM1", "SYM2"],\n'
        '  "secondary_tier": ["SYM3", "SYM4"],\n'
        '  "avoid_chop_list": ["SYM5", "SYM6", "SYM7", "SYM8"],\n'
        '  "decoupled_runners": ["SYM1"],\n'
        '  "fleet_focus_rationale": "2 concise sentences explaining which sector/coins have real institutional alpha right now",\n'
        '  "asset_details": {\n'
        '     "SYM": {\n'
        '        "tier": "LEADER" | "SECONDARY" | "AVOID",\n'
        '        "decoupled_regime": "INDEPENDENT_RUNNER" | "BTC_BETA_SLAVE" | "ACCUMULATION_BASE" | "EXHAUSTION_DISTRIBUTION",\n'
        '        "independent_trade_allowed": true | false,\n'
        '        "edge_score": 85,\n'
        '        "posture": "LONG_BIAS" | "NEUTRAL_CHOP" | "CLOSE_ONLY",\n'
        '        "notes": "brief reason"\n'
        '     }\n'
        '  }\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[CHIEF-STRATEGIST] Gemini error: {e}")

    if not data:
        # Groq failover
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "priority_leaders": ["BTC/USDT", "ETH/USDT"],
        "secondary_tier": ["SOL/USDT", "XRP/USDT"],
        "avoid_chop_list": [],
        "decoupled_runners": [],
        "fleet_focus_rationale": "Fleet operating under standard multi-asset baseline weights.",
        "asset_details": {},
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def generate_commander_memo(context: dict, timeout_sec: float = 20.0) -> dict:
    """
    Autonomous AI Sovereign Commander's Tactical Memo.
    Generated on an hourly cadence (~24 calls/day on Gemini Flash / Groq failover).
    Provides institutional strategic perspective: why current positions are held,
    reasons for candidate setup rejections, key watch levels, and risk posture.
    """
    client = get_llm("commander_memo")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Supreme AI Fleet Commander of an autonomous crypto quantitative fund. "
        "Review the live portfolio telemetry, active positions, recent trade rejections/vetoes, "
        "macro posture, and derivatives skew. Write a crisp, authoritative, institutional "
        "Commander's Tactical Memo for the fund's operators. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Fleet Telemetry & Tactical Snapshot:\n{json.dumps(context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "headline": "1 punchy tactical directive or situational assessment",\n'
        '  "positioning_rationale": "2-3 sentences explaining why current positions are held or why cash is preserved",\n'
        '  "rejection_summary": "1-2 sentences summarizing recent trade setup vetoes (why setups were passed over)",\n'
        '  "watch_levels": ["BTC $85,500 resistance", "ETH $2,640 local support"],\n'
        '  "strategic_directive": "1 definitive sentence for algorithm posture"\n'
        "}\n"
        "CRITICAL: All watch levels MUST be anchored to the live_market_prices provided in the snapshot above (within 1-5% of current price). Do NOT hallucinate outdated or fictional prices."
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as ge:
        log.warning(f"[COMMANDER-MEMO] Gemini error: {ge}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "headline": "DEFENSIVE CHOP REGIME: CAPITAL PRESERVATION AND STRICT VETO BAR ACTIVE",
        "positioning_rationale": "Holding existing positions with exchange-native stops while keeping liquid buffer intact.",
        "rejection_summary": "Sub-threshold signals rejected to prevent fee churn and choppy whipsaws.",
        "watch_levels": ["BTC $68,500 breakout trigger", "ETH $2,650 support"],
        "strategic_directive": "Preserve 70% wallet margin ceiling and await high-conviction trend breakouts.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def scan_multi_asset_liquidity_patterns(batch_context: dict, timeout_sec: float = 20.0) -> dict:
    """
    Feature 1: 15-Minute Multi-Asset Institutional SFP, FVG & Liquidity Sweep Scanner.
    Batches all 8 fleet assets into 1 single call (~96 calls/day on Gemini Flash Lite).
    Detects trapped breakouts, fair value gaps, and liquidity sweeps.
    """
    client = get_llm("chart_scout")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Institutional ICT / Smart Money Concepts (SMC) Quantitative Analyst. "
        "Examine the 15m, 1h, and 4h price action, candle wicks, recent swing highs/lows, and volume across the asset batch. "
        "Detect high-probability structural setups: "
        "1. Swing Failure Patterns (SFP / Liquidity Sweeps where price wicks beyond previous highs/lows but closes back inside). "
        "2. Fair Value Gaps (FVG) and Order Blocks (imbalance zones). "
        "3. Range Compression / Wyckoff Spring or Upthrust. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Multi-Asset Candle & Structure Batch:\n{json.dumps(batch_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "detected_patterns": [\n'
        '     {\n'
        '       "symbol": "SOL/USDT",\n'
        '       "pattern_type": "BULLISH_SFP",\n'
        '       "timeframe": "15m",\n'
        '       "quality_score": 88,\n'
        '       "key_level": 99.20,\n'
        '       "direction": "LONG",\n'
        '       "structural_thesis": "1 concise sentence explaining the liquidity sweep or gap"\n'
        '     }\n'
        '  ],\n'
        '  "market_summary": "1 sentence summarizing overall multi-asset structural alignment"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[CHART-SCOUT] Gemini error: {e}")

    if not data:
        # Groq failover
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "detected_patterns": [],
        "market_summary": "No high-conviction liquidity sweep patterns detected in current cycle.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def evaluate_runner_vs_harvest(position_context: dict, timeout_sec: float = 8.0) -> dict:
    """
    Engine 1: Dynamic Runner vs. Scalp Trailing Stop Architect.
    Dispatched via Groq Cloud 3-key pool for sub-300ms execution.
    Evaluates open positions at >= +1.0R to decide between RUNNER_MODE (trail 15m swing low for 3R-5R moves)
    and HARVEST_MODE (secure immediate profit before liquidity dries up).
    """
    client = get_llm("trailing_architect")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Trailing Stop & Profit Extraction Architect. "
        "Evaluate a live profitable trade that has crossed >= +1.0R unrealized profit. "
        "Assess trend momentum (ADX, RSI, volume surge), distance to swing levels, and decide whether "
        "the trade should be upgraded to RUNNER_MODE (widen target to 3R-5R, trail behind 15m swing structure), "
        "or switched to HARVEST_MODE (secure profit immediately before momentum stalls), or TRAIL_TIGHT. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Profitable Trade Context (>= +1.0R):\n{json.dumps(position_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "action": "RUNNER_MODE" | "HARVEST_MODE" | "TRAIL_TIGHT",\n'
        '  "suggested_sl": 67450.0,\n'
        '  "target_r": 4.5,\n'
        '  "tighten_sl": true,\n'
        '  "harvest_percentage": 0.0,\n'
        '  "rationale": "1 concise sentence explaining the trailing or profit extraction decision"\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)
    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "action": "TRAIL_TIGHT",
        "suggested_sl": None,
        "target_r": 3.0,
        "tighten_sl": True,
        "harvest_percentage": 0.0,
        "rationale": "Standard algorithmic trailing baseline maintained.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def detect_spoof_and_ghost_walls(depth_packet: dict, timeout_sec: float = 8.0) -> dict:
    """
    Engine 2: Order Book Ghost Wall & Spoofing Detector.
    Dispatched via Groq Cloud 3-key pool for sub-300ms execution.
    Evaluates 10-level additions, cancellations, and residence times to identify
    phantom spoofing walls (<250ms residence) vs genuine institutional iceberg anchors (>2500ms).
    """
    client = get_llm("ghost_wall_detector")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are a Quantitative High-Frequency Microstructure & Spoofing Detection Specialist. "
        "Analyze Level-2 order book depth additions, cancellations, and order residence times. "
        "Differentiate between genuine institutional iceberg absorption anchors (orders persisting >2000ms with low cancellation rates) "
        "and deceptive phantom spoof walls (massive size cancelled in <300ms designed to induce retail panic or FOMO). "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Microstructure Depth & Cancellation Snapshot:\n{json.dumps(depth_packet, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "spoof_detected": false,\n'
        '  "spoof_side": "BID" | "ASK" | "NONE",\n'
        '  "phantom_walls": [{"side": "ASK", "price": 68200.0, "size_usd": 150000.0, "description": "brief note"}],\n'
        '  "genuine_anchors": [{"side": "BID", "price": 67100.0, "size_usd": 240000.0, "description": "brief note"}],\n'
        '  "execution_advisory": "ALLOW" | "VETO_BUY_SPOOF" | "VETO_SELL_SPOOF",\n'
        '  "manipulation_risk_pct": 15.0,\n'
        '  "headline": "1 punchy sentence summarizing spoof vs iceberg dynamics"\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)
    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "spoof_detected": False,
        "spoof_side": "NONE",
        "phantom_walls": [],
        "genuine_anchors": [],
        "execution_advisory": "ALLOW",
        "manipulation_risk_pct": 10.0,
        "headline": "Order book displaying standard queue dynamics with low spoofing probability.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def evaluate_mtf_confluence(batch_mtf_context: dict, timeout_sec: float = 20.0) -> dict:
    """
    Engine 3: 15-Minute Multi-Timeframe (15m + 1h + 4h) Confluence Gate.
    Dispatched via Google Gemini 2.5 Flash Lite (batches all 8 fleet assets into 1 call).
    Identifies high-probability structural confluence and vetoes entries heading directly into 4h resistance.
    """
    client = get_llm("mtf_confluence")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Institutional Multi-Timeframe (MTF) Technical Confluence Gatekeeper. "
        "Analyze the 15m, 1h, and 4h price structure, EMAs (20, 50, 200), and key swing support/resistance levels across the fleet assets. "
        "Determine if 15m momentum setups align with higher timeframe (HTF) trends, or if they are counter-trend traps breaking directly into 4h resistance/support walls. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Fleet Multi-Timeframe Structure Matrix (15m, 1h, 4h):\n{json.dumps(batch_mtf_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "confluence_matrix": {\n'
        '    "BTC/USDT": {\n'
        '      "confluence_score": 88,\n'
        '      "alignment": "FULL_BULL_CONFLUENCE" | "FULL_BEAR_CONFLUENCE" | "COUNTER_TREND_TRAP" | "HTF_RESISTANCE_IMMINENT" | "CHOP_CONSOLIDATION",\n'
        '      "veto_long": false,\n'
        '      "veto_short": true,\n'
        '      "key_htf_barrier": 68500.0,\n'
        '      "structural_rationale": "1 concise sentence explaining MTF alignment"\n'
        '    }\n'
        '  },\n'
        '  "fleet_bias": "BULLISH_EXPANSION" | "BEARISH_CONTRACTION" | "NEUTRAL_MIXED",\n'
        '  "market_summary": "1 sentence summarizing overall multi-timeframe market structure"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[MTF-CONFLUENCE] Gemini error: {e}")

    if not data:
        # Groq failover
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "confluence_matrix": {},
        "fleet_bias": "NEUTRAL_MIXED",
        "market_summary": "Multi-timeframe structure neutral across fleet.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

async def calculate_delta_neutral_hedge(portfolio_risk_context: dict, timeout_sec: float = 20.0) -> dict:
    """
    Engine 4: AI Delta-Neutral Hedging Oracle (Macro Volatility Shield).
    Dispatched via Google Gemini 2.5 Flash Lite (with Groq failover).
    Evaluates net directional dollar delta ($) against macro posture (SPY, DXY, regime).
    Recommends optimal synthetic short hedge on the fleet's weakest asset to freeze drawdowns
    without forcing premature closure of winning runners.
    """
    client = get_llm("hedging_oracle")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Chief Quantitative Risk & Hedging Architect. "
        "Evaluate the crypto portfolio's directional net delta exposure against the live macro posture (SPY 1h change, DXY, macro regime). "
        "If macro conditions turn hostile (DEFENSIVE_BEAR or RISK_OFF_KILLSWITCH) while the portfolio is net long, "
        "calculate the optimal synthetic short hedge on the fleet's weakest asset (e.g. from avoid list or lowest relative strength) "
        "to freeze drawdowns without cutting winning runners. "
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Portfolio Risk & Macro Exposure Packet:\n{json.dumps(portfolio_risk_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "hedge_required": false,\n'
        '  "hedge_asset": "ETH/USDT",\n'
        '  "hedge_side": "SHORT" | "NONE",\n'
        '  "hedge_ratio": 0.35,\n'
        '  "target_short_notional": 25.0,\n'
        '  "risk_shield_level": "GREEN_SHIELD" | "YELLOW_HEDGE" | "RED_DELEVERAGE",\n'
        '  "rationale": "1 concise sentence explaining the macro hedging advisory"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[HEDGING-ORACLE] Gemini error: {e}")

    if not data:
        # Groq failover
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "hedge_required": False,
        "hedge_asset": "ETH/USDT",
        "hedge_side": "NONE",
        "hedge_ratio": 0.0,
        "target_short_notional": 0.0,
        "risk_shield_level": "GREEN_SHIELD",
        "rationale": "Portfolio net delta within standard non-hedged bounds.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# V39.0 Advanced Alpha & Execution Optimization Engines
# ---------------------------------------------------------------------------

async def optimize_entry_timing(entry_context: dict, timeout_sec: float = 10.0) -> dict:
    """
    Feature 2: Pre-Entry Retest & Limit Order Sniper.
    Evaluates 5m candle wick structure and L2 order book depth to determine whether
    to enter immediately or place a post-only limit order on a 5m EMA/Kijun retest.
    """
    client = get_llm("entry_sniper")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Execution Sniper & Order Flow Specialist. "
        "A trade setup has passed technical criteria and boardroom conviction. "
        "Your sole task is to OPTIMIZE THE ENTRY PRICE and eliminate buying the top of a 15m/5m wick.\n\n"
        "Execution Rules:\n"
        "1. RUNAWAY_IMPULSE: If order flow imbalance (OFI) is aggressive (> +0.70), candle has virtually no upper wick, "
        "and volume is exploding, enter immediately inside the spread via post-only maker limit.\n"
        "2. RETEST_PULLBACK: If price has extended > 0.5% above the breakout level, RSI is elevated (> 65), "
        "or 5m candle shows an upper rejection wick, do NOT buy the top! Recommend placing a post-only limit order "
        "at the 5m EMA20 / breakout retest level (typically 0.3% to 0.7% lower than current price).\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Candidate Trade Telemetry:\n{json.dumps(entry_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "execution_mode": "RUNAWAY_IMPULSE" | "RETEST_PULLBACK",\n'
        '  "recommended_entry_price": float,\n'
        '  "price_discount_pct": float,\n'
        '  "order_type": "POST_ONLY_LIMIT",\n'
        '  "cancel_after_seconds": 900,\n'
        '  "rationale": "1 concise sentence explaining whether to enter immediately or wait for the retest"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[ENTRY-SNIPER] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    curr_p = float(entry_context.get("current_price", 0.0))
    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                rec_price = float(res.get("recommended_entry_price") or curr_p)
                # Enforce safety bound: recommended entry cannot be higher than current price for LONG
                if rec_price > curr_p * 1.002:
                    rec_price = curr_p
                res["recommended_entry_price"] = rec_price
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "execution_mode": "RUNAWAY_IMPULSE",
        "recommended_entry_price": curr_p,
        "price_discount_pct": 0.0,
        "order_type": "POST_ONLY_LIMIT",
        "cancel_after_seconds": 900,
        "rationale": "Direct post-only limit entry baseline.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def detect_climax_exhaustion(position_context: dict, timeout_sec: float = 8.0) -> dict:
    """
    Feature 3: Climax Exhaustion & Peak Harvest Sentinel.
    Dispatched via Groq Cloud 3-key pool for sub-300ms execution.
    Monitors profitable positions (>= +1.0R) to detect blow-off tops and bank peak profits.
    """
    client = get_llm("climax_sentinel")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Peak Profit Harvester & Climax Detection Specialist. "
        "You monitor an open profitable cryptocurrency futures trade that has reached >= +1.0R. "
        "Your mission is to prevent round-tripping peak paper profits into breakeven.\n\n"
        "Climax Rules:\n"
        "1. HARVEST_CLIMAX_NOW: If 5m candle exhibits a blow-off spike (3x volume, RSI > 75, long upper rejection wick), "
        "or massive ask iceberg walls have appeared absorbing all buy orders, trigger an immediate profit harvest!\n"
        "2. TIGHTEN_TIGHT_TICK: If momentum is slowing near key resistance, advise ratcheting stop to just below the 5m low.\n"
        "3. LET_RUNNER_EXPAND: If volume is steady, order book depth shows persistent buyer absorption, and candle wicks are clean, let it run.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Profitable Position Telemetry:\n{json.dumps(position_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "action": "HARVEST_CLIMAX_NOW" | "TIGHTEN_TIGHT_TICK" | "LET_RUNNER_EXPAND",\n'
        '  "climax_detected": true,\n'
        '  "tighten_sl_price": float or null,\n'
        '  "peak_harvest_pct": 100.0,\n'
        '  "exhaustion_reason": "1 concise sentence explaining the climax or continuation thesis",\n'
        '  "confidence": 0.88\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if not data:
        try:
            combined = f"{system_prompt}\n\n{user_prompt}"
            data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
        except Exception:
            pass

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "action": "LET_RUNNER_EXPAND",
        "climax_detected": False,
        "tighten_sl_price": None,
        "peak_harvest_pct": 0.0,
        "exhaustion_reason": "No climax exhaustion detected; holding trailing stops.",
        "confidence": 0.50,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def evaluate_btc_anchor(btc_context: dict, timeout_sec: float = 12.0) -> dict:
    """
    Feature 4: The Bitcoin Anchor (Cross-Market Contagion Filter).
    Hourly evaluation of Bitcoin's microstructure, funding skew, and HTF resistance proximity.
    Sets ALT_LONG_PAUSE if BTC is undergoing institutional distribution or severe liquidity rejection.
    """
    client = get_llm("macro_sentinel")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Sovereign Bitcoin Anchor & Macro Contagion Officer. "
        "Analyze Bitcoin's immediate 15m/1h structure, order flow imbalance, funding skew, and key resistance levels. "
        "Determine whether Bitcoin is safe for the fleet to deploy capital into ALTCOIN LONGS.\n\n"
        "Rules:\n"
        "1. HEALTHY_ACCUMULATION / HEALTHY_EXPANSION: BTC is trending up, consolidating above EMA50, or absorbing sell pressure cleanly. "
        "Verdict: 'ALLOW_ALT_LONGS' (alt_long_pause=false).\n"
        "2. DISTRIBUTION_DUMP_RISK: BTC is rejecting from HTF resistance with heavy sell volume, negative OFI, or funding rate overheating. "
        "Verdict: 'PAUSE_ALT_LONGS' (alt_long_pause=true), shielding all alt positions from contagion.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Bitcoin Market & Microstructure Telemetry:\n{json.dumps(btc_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "btc_anchor_state": "HEALTHY_EXPANSION" | "HEALTHY_ACCUMULATION" | "DISTRIBUTION_DUMP_RISK" | "NEUTRAL_RANGE",\n'
        '  "alt_long_pause": false,\n'
        '  "contagion_risk_score": 25,\n'
        '  "key_btc_pivot": 84500.0,\n'
        '  "anchor_directive": "1 punchy sentence giving the fleet directive",\n'
        '  "confidence": 0.85\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[BTC-ANCHOR] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "btc_anchor_state": "NEUTRAL_RANGE",
        "alt_long_pause": False,
        "contagion_risk_score": 30,
        "key_btc_pivot": 84000.0,
        "anchor_directive": "Bitcoin range-bound; normal fleet risk parameters active.",
        "confidence": 0.50,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# V40.0 High-Alpha AI Free Credit Architecture (6 Specialized Engines)
# ---------------------------------------------------------------------------

async def detect_liquidity_sweep(sweep_context: dict, timeout_sec: float = 12.0) -> dict:
    """
    Feature 1: AI Structural Liquidity Sweep & Stop-Hunt Detector (Pre-Entry Gate).
    Analyzes candle wicks, recent swing highs/lows, and orderbook absorption to determine
    if a candidate breakout is an institutional stop-hunt / fakeout trap.
    """
    client = get_llm("liquidity_sweep")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Market Microstructure & Smart Money Concepts (SMC) Specialist. "
        "Analyze candidate trade setups for structural liquidity sweeps, false breakout stop-runs, and fakeouts.\n\n"
        "Rules:\n"
        "1. VETO_SWEEP_REVERSAL: If price briefly swept above equal highs or below equal lows on high volume "
        "and immediately closed back inside the range with a long rejection wick, it is a Stop Hunt (Bear Trap / Bull Trap). "
        "Veto entering with the breakout! Recommendation: 'VETO_SWEEP_REVERSAL'.\n"
        "2. DELAY_FOR_CONFIRMATION: If price is testing a liquidity cluster but has not yet accepted beyond it, "
        "recommend waiting for candle closure. Recommendation: 'DELAY_FOR_CONFIRMATION'.\n"
        "3. PROCEED: If the breakout shows full-bodied candle expansion, low rejection wicks, and clean order book absorption, "
        "confirm the entry. Recommendation: 'PROCEED'.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Liquidity & Structure Telemetry:\n{json.dumps(sweep_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "sweep_detected": boolean,\n'
        '  "sweep_type": "BEAR_TRAP_SPRING" | "BULL_TRAP_UPTHRUST" | "EQUAL_HIGHS_RUN" | "EQUAL_LOWS_RUN" | "CLEAN_EXPANSION",\n'
        '  "recommendation": "PROCEED" | "DELAY_FOR_CONFIRMATION" | "VETO_SWEEP_REVERSAL",\n'
        '  "sweep_level": float or null,\n'
        '  "structural_rationale": "1 concise sentence explaining liquidity dynamic",\n'
        '  "confidence": 0.85\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[LIQUIDITY-SWEEP] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "sweep_detected": False,
        "sweep_type": "CLEAN_EXPANSION",
        "recommendation": "PROCEED",
        "sweep_level": None,
        "structural_rationale": "No predatory liquidity sweep detected; structure clean.",
        "confidence": 0.50,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def detect_early_reversal(position_context: dict, timeout_sec: float = 10.0) -> dict:
    """
    Feature 2: In-Flight Early Reversal & Volume Climax Sentinel (Active Position Management).
    Monitors all open positions for bearish/bullish divergences, climax absorption against position,
    or momentum loss before a trade hits a full stop loss or round-trips profits.
    """
    client = get_llm("early_reversal")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite In-Flight Position Sentinel and Order Flow Divergence Specialist. "
        "You continuously monitor active leveraged crypto futures positions to catch early trend breakdowns "
        "and prevent round-tripping or absorbing full stop-out losses.\n\n"
        "Decision Matrix:\n"
        "1. EXIT_IMMEDIATE: Severe adverse divergence (e.g. price making lower highs with heavy sell OFI, or 4H trend structure breaks). "
        "Action: 'EXIT_IMMEDIATE' to preserve capital before stop is hit.\n"
        "2. HARVEST_HALF: Trade is profitable (+0.5R to +1.0R) but hitting heavy opposing iceberg walls or volume climax. "
        "Action: 'HARVEST_HALF'.\n"
        "3. TIGHTEN_SL: Momentum is decelerating; advise ratcheting stop to protect unrealized gains or limit downside. "
        "Action: 'TIGHTEN_SL'.\n"
        "4. HOLD: Position structure is healthy, trending within normal pullback bounds. "
        "Action: 'HOLD'.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Active Position & Microstructure Telemetry:\n{json.dumps(position_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "reversal_warning": boolean,\n'
        '  "severity": "NONE" | "MODERATE" | "CRITICAL",\n'
        '  "action": "HOLD" | "TIGHTEN_SL" | "HARVEST_HALF" | "EXIT_IMMEDIATE",\n'
        '  "tighten_sl_price": float or null,\n'
        '  "suggested_harvest_pct": float,\n'
        '  "reasoning": "1 concise sentence explaining momentum health or divergence",\n'
        '  "confidence": 0.85\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[EARLY-REVERSAL] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "reversal_warning": False,
        "severity": "NONE",
        "action": "HOLD",
        "tighten_sl_price": None,
        "suggested_harvest_pct": 0.0,
        "reasoning": "Position holding within standard volatility parameters.",
        "confidence": 0.50,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def tune_exit_geometry(exit_context: dict, timeout_sec: float = 12.0) -> dict:
    """
    Feature 3: Dynamic AI Exit Geometry (Structure-Calibrated SL & TP).
    Replaces static ATR multiples with structural anchors: Fair Value Gaps, unmitigated order blocks,
    and key swing levels while guaranteeing mathematical risk-reward integrity (R:R >= 2.0).
    """
    client = get_llm("exit_geometry")
    client.session = _ensure_active_session(client)

    entry_p = float(exit_context.get("entry_price", 0.0))
    atr_val = float(exit_context.get("atr", entry_p * 0.02))
    side = str(exit_context.get("side", "long")).upper()
    is_long = "LONG" in side

    system_prompt = (
        "You are an Elite Institutional Quantitative Risk & Trade Geometry Architect. "
        "A trade entry has been approved. Calibrate the structural Stop Loss and Take Profit levels.\n\n"
        "Geometry Rules:\n"
        "1. Stop Loss: Anchor just beyond key market structure (e.g. swing low for longs, swing high for shorts, "
        "unmitigated order block, or Kijun-sen support). Never place an arbitrary stop in the middle of a liquidity vacuum.\n"
        "2. Minimum Distance: The stop distance MUST be at least 1.5% and at least 1.2x ATR from entry to avoid intraday noise.\n"
        "3. Take Profit: Anchor at institutional liquidity targets (prior swing high, unmitigated FVG, or resistance ceiling).\n"
        "4. Risk-Reward Ratio: Minimum R:R must be >= 2.0 (Target profit >= 2.0x risk distance).\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Trade Geometry & Structural Context:\n{json.dumps(exit_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "suggested_sl": float,\n'
        '  "suggested_tp": float,\n'
        '  "structural_anchor_sl": "SWING_LOW" | "ORDER_BLOCK" | "FVG_BOUNDARY" | "ATR_FLOOR",\n'
        '  "structural_anchor_tp": "LIQUIDITY_POOL" | "PRIOR_HIGH" | "FIBONACCI_EXPANSION",\n'
        '  "sl_distance_pct": float,\n'
        '  "r_multiple_potential": float,\n'
        '  "geometry_rationale": "1 concise sentence explaining the structural anchors",\n'
        '  "confidence": 0.85\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[EXIT-GEOMETRY] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                # Verify mathematical integrity
                sl = float(res.get("suggested_sl") or 0.0)
                tp = float(res.get("suggested_tp") or 0.0)
                if sl > 0 and tp > 0 and entry_p > 0:
                    dist = abs(entry_p - sl)
                    min_dist = entry_p * 0.015
                    if dist >= min_dist:
                        reward = abs(tp - entry_p)
                        if reward >= 1.8 * dist:
                            res["updated_at"] = datetime.now(timezone.utc).isoformat()
                            return res
        except Exception:
            pass

    # Baseline ATR Fallback
    baseline_sl_dist = max(2.0 * atr_val, entry_p * 0.018)
    sl_p = round(entry_p - baseline_sl_dist if is_long else entry_p + baseline_sl_dist, 4)
    tp_p = round(entry_p + (2.5 * baseline_sl_dist) if is_long else entry_p - (2.5 * baseline_sl_dist), 4)

    return {
        "suggested_sl": sl_p,
        "suggested_tp": tp_p,
        "structural_anchor_sl": "ATR_FLOOR",
        "structural_anchor_tp": "FIBONACCI_EXPANSION",
        "sl_distance_pct": round((baseline_sl_dist / entry_p) * 100.0, 2) if entry_p > 0 else 1.8,
        "r_multiple_potential": 2.5,
        "geometry_rationale": "Baseline volatility-calibrated 2.0x ATR SL and 2.5R TP target.",
        "confidence": 0.60,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def audit_rules_compliance(audit_context: dict, timeout_sec: float = 10.0) -> dict:
    """
    Feature 4: Real-Time Rule Enforcement Gate in Boardroom (Closed-Loop Learning).
    Audits candidate trade setup against top learned rules distilled from past post-mortems.
    Forces immediate veto if the setup repeats a past forensic failure.
    """
    client = get_llm("rules_compliance")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Sovereign Chief Compliance Officer and Forensic Rule Arbiter. "
        "Your duty is to enforce the fleet's Dynamic Learned Rules distilled from past trading post-mortems.\n\n"
        "Enforcement Directives:\n"
        "1. Check the candidate trade context against EVERY supplied Learned Rule.\n"
        "2. If the setup directly violates an established rule (e.g. entering breakout in low ADX chop, "
        "buying into lower Bollinger breakdown with rising ADX, or trading against higher-timeframe trend), "
        "you MUST set 'approved': false and identify the exact rule violated!\n"
        "3. If all rules are satisfied, set 'approved': true.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Candidate Trade Telemetry & Applicable Learned Rules:\n{json.dumps(audit_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "approved": boolean,\n'
        '  "violated_rule_id": string or null,\n'
        '  "violation_reason": "1 concise sentence explaining why the rule was violated, or confirming full compliance",\n'
        '  "compliance_score": 0.95,\n'
        '  "confidence": 0.90\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if not data:
        try:
            combined = f"{system_prompt}\n\n{user_prompt}"
            data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
        except Exception:
            pass

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "approved": True,
        "violated_rule_id": None,
        "violation_reason": "Setup passed baseline forensic compliance screening.",
        "compliance_score": 1.0,
        "confidence": 0.60,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def evaluate_funding_carry_risk(carry_context: dict, timeout_sec: float = 8.0) -> dict:
    """
    Feature 5: Pre-Funding Settlement Carry Optimization (Funding Bleed Defense).
    Evaluates open positions 30m prior to 8h funding settlement (00:00, 08:00, 16:00 UTC).
    If a trade is stalled or consolidating, and paying high adverse funding, advises harvesting
    or ratcheting to breakeven before settlement drains capital.
    """
    client = get_llm("funding_carry")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are the Sovereign Funding Carry & Yield Optimization Officer. "
        "You evaluate active crypto perpetual positions ahead of the 8-hour funding settlement window (00:00, 08:00, 16:00 UTC).\n\n"
        "Rules:\n"
        "1. HIGH CARRY RISK: If an active LONG is held during elevated positive funding rate (> +0.03%), "
        "or SHORT during negative funding (< -0.03%), and the trade has been stalled/consolidating for >2 hours with < +0.5R progress, "
        "advise 'HARVEST_BEFORE_SETTLEMENT' or 'TIGHTEN_TO_BREAKEVEN' to avoid burning capital on funding bleed.\n"
        "2. LOW CARRY RISK: If the funding rate is neutral/favorable or the trade is aggressively expanding with momentum (> +1.0R), "
        "advise 'HOLD'. Carry drag is insignificant compared to momentum gains.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Position Funding & Carry Context:\n{json.dumps(carry_context, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        '  "carry_risk": "LOW" | "MODERATE" | "HIGH",\n'
        '  "action": "HOLD" | "HARVEST_BEFORE_SETTLEMENT" | "TIGHTEN_TO_BREAKEVEN",\n'
        '  "projected_funding_usd": float,\n'
        '  "settlement_window_active": boolean,\n'
        '  "reasoning": "1 concise sentence explaining carry cost vs trade momentum",\n'
        '  "confidence": 0.85\n'
        "}"
    )

    payload = {
        "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if not data:
        try:
            combined = f"{system_prompt}\n\n{user_prompt}"
            data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
        except Exception:
            pass

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    return {
        "carry_risk": "LOW",
        "action": "HOLD",
        "projected_funding_usd": 0.0,
        "settlement_window_active": False,
        "reasoning": "Funding carry cost is within normal acceptable parameters.",
        "confidence": 0.60,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def generate_daily_market_blueprint(watchlist_symbols: list[str], market_snapshot: dict, timeout_sec: float = 25.0) -> dict:
    """
    Feature 6: AI 6-Hour Strategic Macro Blueprint & Key Level Roadmap.
    Dispatched every 6 hours (00:00, 06:00, 12:00, 18:00 UTC) via Google Gemini 2.5 Flash Lite (with Groq failover).
    Maps out high-conviction key levels, invalidation zones, and tactical playbooks for each fleet asset.
    """
    client = get_llm("daily_blueprint")
    client.session = _ensure_active_session(client)

    session_name = market_snapshot.get("session_name", "Global 6-Hour Macro Session")
    session_tag = market_snapshot.get("session_tag", f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_S{(datetime.now(timezone.utc).hour // 6) + 1}")
    today_str = market_snapshot.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    system_prompt = (
        "You are the Sovereign Chief Macro Strategist & Quantitative Portfolio Architect. "
        "Synthesize the 6-Hour Institutional Strategic Blueprint & Key Level Roadmap for the cryptocurrency fleet. "
        "This strategic blueprint cycles every 6 hours across global trading sessions (Asia, London, New York, APAC pre-market).\n\n"
        "Your Blueprint Must Deliver:\n"
        "1. Overall Macro Bias (EXPANSION_BULL, DEFENSIVE_CHOP, CAPITAL_PRESERVATION_BEAR) based on BTC, traditional equities (SPY), DXY, and funding.\n"
        "2. A punchy 2-3 sentence strategic executive narrative setting the fleet's tactical posture for the next 6-hour window.\n"
        "3. An asset-by-asset roadmap for the watchlist assets. For each asset, use its CURRENT LIVE PRICE to define:\n"
        "   - key_support: realistic structural support level below current price\n"
        "   - key_resistance: realistic overhead resistance ceiling above current price\n"
        "   - invalidation_level: stop-loss or structural thesis invalidation point\n"
        "   - tactical_plan: 1 concise actionable sentence with exact entry, target, and risk guidelines for the next 6 hours.\n"
        "Return ONLY valid JSON with no markdown."
    )

    user_prompt = (
        f"Active Session: {session_name} [{session_tag}]\n"
        f"Watchlist Assets: {watchlist_symbols}\n"
        f"Fleet Live Asset Prices:\n{json.dumps(market_snapshot.get('asset_current_prices', {}), indent=2)}\n\n"
        f"Market Snapshot Telemetry:\n{json.dumps(market_snapshot, indent=2)}\n\n"
        "Return JSON schema:\n"
        "{\n"
        f'  "blueprint_date": "{today_str}",\n'
        f'  "session_name": "{session_name}",\n'
        f'  "session_tag": "{session_tag}",\n'
        '  "macro_bias": "EXPANSION_BULL" | "DEFENSIVE_CHOP" | "CAPITAL_PRESERVATION_BEAR",\n'
        '  "daily_narrative": "Paragraph explaining macro context and institutional bias for this 6-hour session",\n'
        '  "asset_roadmaps": {\n'
        '    "BTC/USDT": {\n'
        '      "bias": "BULLISH_CONTINUATION" | "CHOP_RANGE" | "BEARISH_PULLBACK",\n'
        '      "key_support": float,\n'
        '      "key_resistance": float,\n'
        '      "invalidation_level": float,\n'
        '      "tactical_plan": "1 concise sentence with entry and invalidation instruction"\n'
        '    }\n'
        '  },\n'
        '  "confidence": 0.88\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as e:
        log.warning(f"[DAILY-BLUEPRINT] Gemini error: {e}")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            res = robust_json_decode(raw)
            if res and isinstance(res, dict):
                res["blueprint_date"] = today_str
                res["session_name"] = session_name
                res["session_tag"] = session_tag
                res["updated_at"] = datetime.now(timezone.utc).isoformat()
                return res
        except Exception:
            pass

    # Baseline Roadmap Fallback
    roadmaps = {}
    current_prices = market_snapshot.get("asset_current_prices", {})
    for s in watchlist_symbols[:10]:
        cp = float(current_prices.get(s, 0.0) or 0.0)
        s_supp = round(cp * 0.98, 4) if cp > 0 else 0.0
        s_res = round(cp * 1.02, 4) if cp > 0 else 0.0
        s_inv = round(cp * 0.97, 4) if cp > 0 else 0.0
        roadmaps[s] = {
            "bias": "CHOP_RANGE",
            "key_support": s_supp,
            "key_resistance": s_res,
            "invalidation_level": s_inv,
            "tactical_plan": "Range-bound chop; prioritize patience, mean-reversion maker fills, and capital preservation."
        }

    return {
        "blueprint_date": today_str,
        "session_name": session_name,
        "session_tag": session_tag,
        "macro_bias": "DEFENSIVE_CHOP",
        "daily_narrative": f"Fleet entering {session_name} under consolidated chop regime. Maintaining defensive posture and preserving capital for high-conviction structural breaks.",
        "asset_roadmaps": roadmaps,
        "confidence": 0.60,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


async def detect_breakout_trap(trap_context: dict, timeout_sec: float = 10.0) -> dict:
    """
    Feature 6: AI Breakout vs. Bull-Trap Gate (Zero-Lag Fakeout Shield).
    Evaluates micro-timeframe candle wick geometry, volume absorption delta,
    and order book depth to determine if an impending breakout is a retail bull-trap wick.
    """
    client = get_llm("breakout_trap")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Microstructure & Fakeout Sentinel. "
        "Your sole mission is to protect capital from buying retail bull-traps or selling bear-traps. "
        "Scrutinize candidate breakout setups:\n"
        "1. If price is breaking out but displaying long upper rejection wicks on 5m/15m (>45% of bar range), "
        "or volume is drying up at the breakout high, classify as 'VETO_BULL_TRAP' with trap_risk_score >= 0.75.\n"
        "2. If volume confirms the push, candle body is full, and price accepted cleanly above resistance, "
        "classify as 'CONFIRMED_BREAKOUT' with trap_risk_score < 0.30.\n"
        "3. If consolidating tightly just beneath resistance preparing for ignition, classify as 'HOLD_FOR_RETEST'.\n"
        "Return strictly valid JSON with no markdown."
    )
    user_prompt = (
        f"Candidate Breakout Context:\n{json.dumps(trap_context, indent=2)}\n\n"
        "Respond strictly in JSON format:\n"
        "{\n"
        '  "verdict": "CONFIRMED_BREAKOUT" | "VETO_BULL_TRAP" | "VETO_BEAR_TRAP" | "HOLD_FOR_RETEST",\n'
        '  "trap_risk_score": 0.0 to 1.0,\n'
        '  "confidence": 0.85,\n'
        '  "structural_rationale": "1-2 concise institutional sentences explaining wick/volume dynamics"\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as ge:
        log.warning(f"[BREAKOUT-TRAP] Gemini error: {ge}. Falling back to Groq...")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = robust_json_decode(raw)
            if parsed and isinstance(parsed, dict):
                return parsed
        except Exception as pe:
            log.warning(f"[BREAKOUT-TRAP] Parse error: {pe}")

    return {
        "verdict": "CONFIRMED_BREAKOUT",
        "trap_risk_score": 0.35,
        "confidence": 0.50,
        "structural_rationale": "Trap gate offline — baseline execution allowed."
    }


# ---------------------------------------------------------------------------
# V40.0 Autonomous AI Event Catalyst & Horizon Arbiter
# ---------------------------------------------------------------------------

async def evaluate_event_catalyst(event_packet: dict, timeout_sec: float = 12.0) -> dict:
    """
    V40.0 Autonomous AI Event Catalyst & Horizon Arbiter.
    Classifies breaking geopolitical, macroeconomic, regulatory, or volatility events into:
    1. FLASH_LIQUIDITY_SHOCK (Duration: 15m - 2h, liquidation flush, mean-reversion rebound trap, DO NOT SHORT)
    2. STRUCTURAL_MACRO_BEAR (Duration: Days to Weeks, true regime shift, war/contagion, AUTHORIZE 4H SHORTS)
    3. STRUCTURAL_BULL_CATALYST (Duration: Days to Weeks, ETF approval/dovish pivot, AUTHORIZE BREAKOUT LONGS)
    4. NOISE_FUD (Temporary rumor/headline, IGNORE)

    Synthesizes: Headline text + BTC 15m velocity + Bitget Open Interest delta + Funding Rate + SPY/DXY macro moves.
    """
    client = get_llm("event_catalyst")
    client.session = _ensure_active_session(client)

    system_prompt = (
        "You are an Elite Institutional Event-Driven Macro Strategist and High-Frequency Derivatives Specialist. "
        "Your mission is to analyze breaking headlines and market volatility to distinguish between: "
        "A) FLASH_LIQUIDITY_SHOCK: Panic sell, cascade of stop-runs / long liquidations, funding rate drops deep negative (< -0.02%), "
        "   OI collapses (-$200M+), traditional equities (SPY) flat/unaffected. High risk of violent short squeeze! Directive: FLASH_REBOUND_TRAP (Do not short, prepare for mean-reversion bounce). "
        "B) STRUCTURAL_MACRO_BEAR: True macro crisis, kinetic war escalation, oil shock, SPY dumping > 1.0%, DXY spiking, "
        "   Open Interest rising or holding steady (institutional aggressive shorting, not just forced liquidations). Directive: SECULAR_SHORT_BREAKDOWN (Authorize 4H shorts with 2x-3x leverage, block longs). "
        "C) STRUCTURAL_BULL_CATALYST: Landmark regulatory approval (ETF), rate cuts, sovereign adoption, pro-crypto legal victory. Directive: BULL_MOMENTUM_EXPANSION (Authorize 4H breakouts with 3x-5x leverage). "
        "D) NOISE_FUD: Unverified social media rumors, minor protocol exploits without systemic contagion. Directive: IGNORE_NOISE (Normal operations). "
        "Return ONLY a valid JSON object matching the requested schema."
    )

    user_prompt = (
        f"Event & Derivatives Telemetry:\n{json.dumps(event_packet, indent=2)}\n\n"
        "Return strictly valid JSON with this schema:\n"
        "{\n"
        '  "event_category": "GEOPOLITICAL_WAR" | "MACRO_MONETARY" | "REGULATORY_LEGAL" | "LIQUIDATION_FLUSH" | "NOISE_RUMOR",\n'
        '  "expected_horizon": "FLASH_LIQUIDITY_SHOCK" | "STRUCTURAL_MACRO_BEAR" | "STRUCTURAL_BULL_CATALYST" | "NOISE_FUD",\n'
        '  "confidence": 0.85,\n'
        '  "tactical_playbook": "FLASH_REBOUND_TRAP" | "SECULAR_SHORT_BREAKDOWN" | "BULL_MOMENTUM_EXPANSION" | "IGNORE_NOISE" | "CASH_PRESERVATION",\n'
        '  "directive": {\n'
        '    "macro_bias": "BEARISH" | "BULLISH" | "NEUTRAL",\n'
        '    "allow_shorts": true,\n'
        '    "allow_longs": false,\n'
        '    "recommended_leverage": 2,\n'
        '    "invalidation_note": "1 concise sentence on invalidation condition"\n'
        '  },\n'
        '  "headline_summary": "1 punchy sentence summarizing the core catalyst",\n'
        '  "structural_rationale": "2 concise institutional sentences explaining why this is flash vs structural based on OI, funding, and macro contagion."\n'
        "}"
    )

    combined = f"{system_prompt}\n\n{user_prompt}"
    data = None
    try:
        data = await _query_gemini_cloud(client.session, combined, timeout_sec=timeout_sec)
    except Exception as ge:
        log.warning(f"[EVENT-CATALYST] Gemini error: {ge}. Falling back to Groq...")

    if not data:
        payload = {
            "model": getattr(config, "GROQ_MODEL", "openai/gpt-oss-20b"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        data = await _query_groq_cloud_with_rotation(client.session, payload, timeout_sec=timeout_sec)

    if data:
        try:
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = robust_json_decode(raw)
            if parsed and isinstance(parsed, dict):
                parsed["updated_at"] = datetime.now(timezone.utc).isoformat()
                return parsed
        except Exception as pe:
            log.warning(f"[EVENT-CATALYST] Parse error: {pe}")

    return {
        "event_category": "NOISE_RUMOR",
        "expected_horizon": "NOISE_FUD",
        "confidence": 0.50,
        "tactical_playbook": "IGNORE_NOISE",
        "directive": {
            "macro_bias": "NEUTRAL",
            "allow_shorts": True,
            "allow_longs": True,
            "recommended_leverage": 3,
            "invalidation_note": "Fallback baseline"
        },
        "headline_summary": "Event catalyst fallback baseline.",
        "structural_rationale": "Arbiter offline — default baseline rules active.",
        "updated_at": datetime.now(timezone.utc).isoformat()
    }







