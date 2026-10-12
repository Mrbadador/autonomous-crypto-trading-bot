import re
import os
import sys

def refactor():
    filename = 'berserker_trader.py'
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Refactor setup_logging to avoid global force=True reset
    new_setup = """
def setup_logging(log_name="berserker"):
    l = logging.getLogger(log_name)
    if not l.handlers:
        l.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        l.addHandler(sh)
        
        os.makedirs("logs", exist_ok=True)
        fh = logging.FileHandler(f"logs/{log_name}.log")
        fh.setFormatter(formatter)
        l.addHandler(fh)
    l.propagate = False # Prevent duplicate bubbling
    return l
"""
    old_setup_pat = r'def setup_logging\(log_name="berserker"\):.*?return log'
    content = re.sub(old_setup_pat, new_setup.strip(), content, flags=re.DOTALL)

    # 2. Refactor __init__ signature and content
    # Extract current init to keep other variable setup
    init_block_pattern = r'class BerserkerTrader:.*?def __init__\(self, log_name="berserker"\):'
    
    new_init_header = """class BerserkerTrader:
    def __init__(self, symbol: str, shared_warden=None, shared_sentinel=None):
        prefix = symbol.split('/')[0].upper()
        self.log_name = f"berserker-{prefix}"
        self.log = setup_logging(self.log_name)
        self.log.info(f"[BOOT] Fleet Node for {symbol} initializing.")
        
        self.symbol = symbol
        self.symbols = [symbol]
        self.timeframe = "5m"
        self.exchange = None
        self.paper_ex = LivePaperExchange(initial_balance=config.INITIAL_BALANCE, account_name="berserker")
        self.dfs = {s: pd.DataFrame() for s in self.symbols}
        self.active_positions = {s: None for s in self.symbols}
        self.last_btc_trend = "UNKNOWN"
        self.order_lock = asyncio.Lock()
        self.veto_pause_until = {}
        self.tp_cooldown = {}
        self.last_macro_fetch = None
        self.macro_fetch_ttl = timedelta(minutes=10)
        self.state = EngineState()

        self._last_ticker: dict[str, dict] = {}
        self._last_balance: float = 0.0
        self._balance_fetched_at: float = 0.0
        self._balance_ttl: float = 10.0
        
        self.rl_analyst = AnalystAgent()
        self.rl_strategist = StrategistAgent()
        
        self.macro_warden = shared_warden if shared_warden else Warden60Minute(db_path="trading_ledger.db")
        self.news_sentinel = shared_sentinel if shared_sentinel else news_sentinel
"""
    # Regex to match class def and the standard vars down to Warden definition
    regex_to_replace = r'class BerserkerTrader:.*?def __init__\(self, log_name="berserker"\):.*?self\.macro_warden = Warden60Minute\(db_path="trading_ledger\.db"\)'
    
    content = re.sub(regex_to_replace, new_init_header.strip(), content, flags=re.DOTALL)

    # 3. Convert global references of 'log.' to 'self.log.' within the class structure
    # We only target lines indented with 4 or more spaces that invoke 'log.'
    content = re.sub(r'^([ \t]{4,})log\.', r'\1self.log.', content, flags=re.MULTILINE)

    # 4. Convert occurrences of news_sentinel global to self.news_sentinel inside the class methods
    content = re.sub(r'^([ \t]{4,}.*?)news_sentinel\.', r'\1self.news_sentinel.', content, flags=re.MULTILINE)

    # 5. Update websocket_loop URL to be dynamic based on the symbol
    old_ws = r'url = "wss://fstream\.binance\.com/market/stream\?streams=solusdt@kline_5m/btcusdt@kline_5m"'
    new_ws = r'sym_raw = self.symbol.replace("/", "").lower()\n        url = f"wss://fstream.binance.com/market/stream?streams={sym_raw}@kline_5m"'
    content = re.sub(old_ws, new_ws, content)

    # 6. Fix collision points on update_system_state
    # current_status_berserker -> current_status_{self.log_name}
    content = content.replace('"current_status_berserker"', 'f"current_status_{self.log_name}"')
    content = content.replace('"last_veto_berserker"', 'f"last_veto_{self.log_name}"')
    # berserker_metrics -> {self.log_name}_metrics
    content = content.replace('"berserker_metrics"', 'f"{self.log_name}_metrics"')

    # 7. Update metrics write near line 287
    # (Handled by step 6 replacement if it finds "berserker_metrics")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Refactor Successful: BerserkerTrader rewritten for fleet concurrency.")

if __name__ == "__main__":
    refactor()
