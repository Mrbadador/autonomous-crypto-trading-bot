import os
import json
import requests
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from email_dispatcher import send_whale_warning

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] WHALE_WATCHER: %(message)s")
log = logging.getLogger("whale_watcher")

# 1. Environment Loading
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq/compound-mini"

if not API_KEY:
    log.error("GROQ_API_KEY missing from .env. Graceful exit.")
    sys.exit(0)

# 2. Data Source (Scraping Whale Movements)
def fetch_whale_movements():
    # Using a scrape-friendly public whale tracker as a source
    # ClankApp or similar trackers often have clean HTML lists
    url = "https://clankapp.com/tx/btc" 
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        log.info(f"Scraping whale movements from {url}...")
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        # Simple extraction logic: Find text that looks like transaction descriptions
        # In a real scenario, we'd use BeautifulSoup, but for 'Lightweight Dispatcher' 
        # we'll use string splitting or regex to find common patterns.
        text = response.text
        
        # Look for transaction snippets (very common in whale trackers)
        # e.g., "Unknown wallet to Binance", "1,200 BTC transferred"
        # We'll grab the first 5000 characters of the body to keep the Groq payload small
        start_marker = "<tbody>"
        end_marker = "</tbody>"
        if start_marker in text and end_marker in text:
            raw_data = text.split(start_marker)[1].split(end_marker)[0]
        else:
            raw_data = text[:5000] # Fallback to top of page
            
        return raw_data
    except Exception as e:
        log.error(f"Failed to scrape whale data: {e}")
        return ""

# 3. Groq Analysis (Mega-Move Filter)
def analyze_whale_threats(raw_html):
    if not raw_html or len(raw_html) < 100:
        log.warning("Insufficient whale data to analyze.")
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = (
        "You are an On-Chain Analyst. Look for movements from 'Unknown Wallet' to 'Exchange' (Binance, Coinbase, etc). "
        "If a single move is > $50,000,000 or > 1,000 BTC or > 100,000 SOL, set status to 'DANGER'. "
        "Otherwise, set to 'CALM'. "
        "Output exactly 1 JSON object with keys: 'whale_status' (CALM|DANGER), 'last_mega_move' (Short description), 'impact_level' (1-10)."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract and analyze whale moves from this HTML snippet:\n{raw_html[:3000]}"}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        log.info("Requesting on-chain threat analysis from Groq...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        log.error(f"Groq Whale Analysis Failed: {e}")
        return None

# 4. State Persistence
def save_whale_state(analysis):
    if not analysis:
        return

    state_file = "whale_state.json"
    try:
        final_state = {
            "whale_status": analysis.get("whale_status", "CALM"),
            "last_mega_move": analysis.get("last_mega_move", "None detected"),
            "impact_level": analysis.get("impact_level", 1),
            "last_update": datetime.now().isoformat()
        }
        
        with open(state_file, "w") as f:
            json.dump(final_state, f, indent=4)
        
        log.info(f"Whale Watcher updated: Status={final_state['whale_status']} | Impact={final_state['impact_level']}/10")
        
        # V16.2: Dispatch Emergency Email if DANGER detected
        if final_state["whale_status"] == "DANGER":
            send_whale_warning(final_state["last_mega_move"], final_state["impact_level"])
    except Exception as e:
        log.error(f"Failed to save whale state: {e}")

# 5. Main Execution
if __name__ == "__main__":
    log.info("--- Initiating Whale Watcher Audit ---")
    
    # Step A: Scrape
    data_snippet = fetch_whale_movements()
    
    # Step B: Filter via Groq
    whale_analysis = analyze_whale_threats(data_snippet)
    
    # Step C: Persist
    save_whale_state(whale_analysis)
    
    log.info("Whale Audit Complete.")
