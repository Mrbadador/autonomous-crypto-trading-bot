import os
import sqlite3
import json
import requests
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from email_dispatcher import send_morning_briefing

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] QUANT_ANALYST: %(message)s")
log = logging.getLogger("quant_analyst")

# 1. Environment Loading
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "groq/compound-mini"

if not API_KEY:
    log.error("GROQ_API_KEY missing from .env. Graceful exit.")
    sys.exit(0)

# 2. Local Math (Pre-Processor)
def get_trade_summary(limit=50):
    db_path = "trading_data.db"
    if not os.path.exists(db_path):
        log.error(f"Database {db_path} not found.")
        return None

    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Fetch last 50 completed trades
        cursor.execute("SELECT symbol, pnl, side FROM trades WHERE status = 'closed' ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        trades = [dict(row) for row in rows]
        total = len(trades)
        wins = [t for t in trades if float(t['pnl']) > 0]
        losses = [t for t in trades if float(t['pnl']) < 0]
        
        win_rate = (len(wins) / total * 100) if total > 0 else 0
        avg_win = (sum(float(w['pnl']) for w in wins) / len(wins)) if wins else 0
        avg_loss = (sum(float(l['pnl']) for l in losses) / len(losses)) if losses else 0
        
        # Asset-specific win rates
        def calc_sym_winrate(sym):
            sym_trades = [t for t in trades if sym.upper() in t['symbol'].upper()]
            if not sym_trades: return 0.0
            sym_wins = [t for t in sym_trades if float(t['pnl']) > 0]
            return (len(sym_wins) / len(sym_trades)) * 100

        summary = {
            "total_trades": total,
            "win_rate_pct": round(win_rate, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "btc_win_rate_pct": round(calc_sym_winrate("BTC"), 2),
            "sol_win_rate_pct": round(calc_sym_winrate("SOL"), 2)
        }
        
        log.info(f"Processed summary for last {total} trades.")
        return summary
    except Exception as e:
        log.error(f"Local Math failed: {e}")
        return None

# 3. Micro-Payload Dispatch
def analyze_summary(summary):
    if not summary:
        return None

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # The Micro-Payload string
    summary_str = (
        f"Analyze this summary data: {summary['total_trades']} trades, {summary['win_rate_pct']}% win rate, "
        f"Avg Win ${summary['avg_win']}, Avg Loss ${summary['avg_loss']}. "
        f"BTC win rate {summary['btc_win_rate_pct']}%, SOL win rate {summary['sol_win_rate_pct']}%."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system", 
                "content": (
                    "You are a world-class Quant Analyst. Review the summary trading data. "
                    "Output exactly 3 actionable trading rules in strict JSON format under the key 'rules'. "
                    "Example: {'rules': ['Rule 1', 'Rule 2', 'Rule 3']}"
                )
            },
            {"role": "user", "content": summary_str}
        ],
        "response_format": {"type": "json_object"}
    }

    try:
        log.info("Dispatching Micro-Payload to Groq (Llama 3.1)...")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        res_json = response.json()
        content = res_json['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        log.error(f"Groq Dispatch Failed: {e}")
        return None

# 4. Safe Memory Update
def update_learned_memory(analysis):
    if not analysis or "rules" not in analysis:
        return

    memory_file = "learned_memory.json"
    try:
        memory = {}
        if os.path.exists(memory_file):
            with open(memory_file, "r") as f:
                try:
                    memory = json.load(f)
                except:
                    pass

        memory["learned_rules"] = analysis["rules"]
        memory["last_audit_date"] = datetime.now().isoformat()
        
        with open(memory_file, "w") as f:
            json.dump(memory, f, indent=4)
        
        log.info(f"Learned Memory Updated. {len(analysis['rules'])} new rules active.")
        
        # V16.2: Dispatch Email Notification
        send_morning_briefing(analysis["rules"])
    except Exception as e:
        log.error(f"Memory Update Failed: {e}")

# 5. Execution
if __name__ == "__main__":
    log.info("--- Initiating Micro-Payload Quant Analysis ---")
    
    # Step A: Local Math
    summary_data = get_trade_summary(50)
    
    if summary_data:
        # Step B: Query Groq
        analysis_result = analyze_summary(summary_data)
        
        if analysis_result:
            # Step C: Update
            update_learned_memory(analysis_result)
    
    log.info("Analysis Cycle Complete.")
