import os
import json
import requests
import sys
import logging
import feedparser
import time
import tempfile
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SENTIMENT_ORACLE: %(message)s")
log = logging.getLogger("sentiment_oracle")

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq/compound-mini"
STATE_FILE = "sentiment_state.json"

def get_last_known_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Could not read fallback state: {e}")
    
    return {
        "BTC": {"status": "SAFE", "score": 5, "reason": "Fallback default"},
        "SOL": {"status": "SAFE", "score": 5, "reason": "Fallback default"},
        "fear_greed_index": 0.5,
        "last_update": datetime.now().isoformat()
    }

def fetch_fear_and_greed():
    """Fetch Bitcoin Fear & Greed Index from alternative.me and scale 0.0 to 1.0"""
    try:
        log.info("Fetching Fear & Greed Index from alternative.me...")
        response = requests.get("https://api.alternative.me/fng/", timeout=3.0)
        response.raise_for_status()
        data = response.json()
        fng_val = float(data["data"][0]["value"])
        scaled_fng = fng_val / 100.0  # Scale explicitly to 0.0 - 1.0
        log.info(f"F&G Index: {fng_val} (Scaled: {scaled_fng:.4f})")
        return scaled_fng
    except requests.exceptions.RequestException as e:
        log.warning(f"F&G API network exception/timeout: {e}")
    except Exception as e:
        log.warning(f"F&G Parsing failure: {e}")
    return None

def fetch_headlines():
    """Fetch Crypto News Sentiment Stream from RSS"""
    feeds = [
        "https://cointelegraph.com/rss/tag/bitcoin",
        "https://cointelegraph.com/rss/tag/solana",
        "https://www.coindesk.com/arc/outboundfeeds/rss/"
    ]
    all_headlines = []
    for url in feeds:
        try:
            log.info(f"Fetching news feed: {url}")
            resp = requests.get(url, timeout=3.0)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:10]:
                tags = entry.get('tags', [{'term': 'News'}])
                term = tags[0]['term'] if tags else 'News'
                all_headlines.append(f"[{term}] {entry.title}")
        except requests.exceptions.RequestException as e:
            log.warning(f"Feed {url} network timeout/exception: {e}")
        except Exception as e:
            log.warning(f"Feed {url} parsing failure: {e}")

    unique_headlines = list(set(all_headlines))
    return "\n".join(unique_headlines[:20])

def get_sentiment_analysis(headlines):
    """Compute rolling hourly text sentiment baseline via Groq"""
    if not headlines:
        log.warning("No headlines fetched, cannot compute text sentiment.")
        return None
    
    if not API_KEY:
        log.warning("GROQ_API_KEY missing, skipping LLM analysis.")
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are a Crypto Risk Manager. Analyze these headlines. "
        "Calculate a rolling hourly text sentiment baseline. "
        "Output strictly 2 objects (BTC, SOL) in valid JSON format. "
        "Each object must have 'status' (SAFE|WARNING|BLOCK), 'score' (1-10, where 10 is most bullish), and 'reason'. "
        "If news is severely bearish/hack, set BLOCK. If just bearish, WARNING. Otherwise SAFE."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze these headlines:\n{headlines}"}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        log.info("Requesting LLM Sentiment baseline...")
        response = requests.post(url, headers=headers, json=payload, timeout=3.0)
        response.raise_for_status()
        
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        return json.loads(content)
    except requests.exceptions.RequestException as e:
        log.warning(f"Groq API network exception/timeout: {e}")
    except Exception as e:
        log.warning(f"Groq Parsing failure: {e}")
    return None

def save_sentiment_state(new_state):
    """Write payload using tempfile atomic swap to prevent corruption"""
    try:
        dir_path = os.path.dirname(os.path.abspath(STATE_FILE))
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(new_state, f, indent=4)
        
        os.replace(temp_path, STATE_FILE)
        log.info("Successfully atomic-swapped sentiment_state.json")
    except Exception as e:
        log.error(f"Failed atomic state write: {e}")

if __name__ == "__main__":
    import asyncio
    from sentinel_agent import SentimentSentinel, FLEET_TICKERS

    log.info("--- Initiating Unified Fleet Sentiment Oracle (V43.0) ---")
    start_time = time.perf_counter()

    sentinel = SentimentSentinel()
    ok = asyncio.run(sentinel.refresh_once(FLEET_TICKERS))
    exec_latency = time.perf_counter() - start_time

    if not ok:
        log.error("Oracle refresh failed (0 headlines). Preserving old timestamp so stale state ages out fail-closed.")
        sys.exit(1)

    final_state = get_last_known_state()
    fng_value = float(final_state.get("fear_greed_index", 0.5))
    btc_score = float(final_state.get("BTC", {}).get("score", 8.0))

    tensor_n_fg = max(-1.0, min(1.0, (fng_value - 0.5) * 2.0))
    tensor_n_sen = max(-1.0, min(1.0, (btc_score - 5.5) / 4.5))

    log.info(f"Unified Oracle Cycle Complete in {exec_latency:.2f}s | Tickers: {len(FLEET_TICKERS)} + GLOBAL | F&G Raw: {final_state.get('fear_greed_raw')}")
    print("\n" + "=" * 50)
    print("      LIVE REST EXTRACTION METRICS (NORMALIZED)")
    print("=" * 50)
    print(f"[{tensor_n_fg:.4f}, {tensor_n_sen:.4f}]")
    print("=" * 50 + "\n")
