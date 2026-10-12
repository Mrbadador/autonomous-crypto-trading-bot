from datetime import datetime, UTC, timedelta
from typing import Optional
import json
import os
import csv
import logging
import atexit
from dataclasses import dataclass, asdict
from db_manager import DatabaseManager
from notifications import NotificationManager
import config

log = logging.getLogger(__name__)

@dataclass
class TradeEntry:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    balance: float
    ai_action: str
    final_action: str
    ai_notes: str
    regime: str
    phase: str
    quality: float
    sl_mode: str
    tp_mode: str
    trail_mode: str
    timestamp: datetime
    mfe: float = 0.0
    pos_id: Optional[int] = None

class TradeJournal:
    def __init__(self, filename="trades.csv"):
        self.filename = filename
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "side", "entry_price", "exit_price", 
                    "qty", "pnl", "ai_action", "final_action", "phase", "ai_notes"
                ])

    def add(self, entry: TradeEntry):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                entry.timestamp.isoformat(), entry.symbol, entry.side, 
                entry.entry_price, entry.exit_price, entry.qty, 
                entry.pnl, entry.ai_action, entry.final_action, 
                entry.phase, entry.ai_notes
            ])

class EngineState:
    def __init__(self, sim_instance=None):
        self.db = DatabaseManager()
        self.notifier = NotificationManager()
        self.journal = TradeJournal()
        self.sim = sim_instance
        
        # Portfolio Metrics — SRE-P0 Fix: Restore true historical peak from SQLite
        self.start_balance = self._restore_start_balance()
        self.max_balance = self._restore_peak_balance()
        self.max_drawdown = 0.0

        # Operational Safeguards
        self.is_halted = False
        self.halt_logged = False
        self.is_observe_only = False
        self.llm_failures = 0
        self.global_streak = 0
        self.global_trade_timestamps = []
        self.global_btc_trend = "unknown"
        self.in_position = {} # dict symbol -> bool
        self.active_trade = {} # dict symbol -> Pos object
        self.cycle_count = 0
        self.last_llm_failure = None

        # Per-Symbol Tracking
        self.consecutive_sl_hits = {}
        self.cooldown_until = {}
        self.last_candle_times = {}
        self.symbol_pnl_history = {} # symbol -> [(datetime, pnl), ...]
        self.atr_history = {}
        self.last_price = {}
        self.last_atr = {}
        self.vol_regime_streak = {}
        self.last_funding_fetch = {}
        self.last_funding_log = {}
        self.last_vol_log = {}
        self.last_news_log = {}

        # Telemetry
        self.last_update = datetime.now(UTC)
        self.post_trade_notes = []
        self.upcoming_news = []
        self.news_last_fetch = None

        # File-based logging
        self.csv_file = "trade_history.csv"
        self._init_csv()
        
        # Guarantee cleanup on shutdown
        atexit.register(self.close)

    def _init_csv(self):
        if not os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp", "symbol", "type", "side", "entry", "exit", "qty", "pnl", "balance",
                        "regime", "phase", "quality", "sl_mode", "tp_mode", "trail_mode", "mfe", "notes", "pos_id"
                    ])
            except Exception as e:
                log.warning(f"Error initializing {self.csv_file}: {e}")

    def close(self):
        """Cleanup resources."""
        if hasattr(self, 'notifier') and self.notifier:
            try: 
                self.notifier.stop()
            except Exception as e: 
                log.warning(f"Error stopping notifier: {e}")

    def _restore_start_balance(self) -> float:
        """Restores canonical user deposit baseline from system_state."""
        try:
            import sqlite3
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                row = conn.execute("SELECT value FROM system_state WHERE key = 'user_total_deposit'").fetchone()
                if row and row[0]:
                    return float(row[0])
        except Exception:
            pass
        return 250.0

    def _restore_peak_balance(self) -> float:
        """SRE-P0 Fix: Restores historical High-Water Mark from system_state to prevent HWM amnesia on reboot."""
        peaks = [self.start_balance]
        try:
            import sqlite3
            with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                rows = conn.execute("SELECT value FROM system_state WHERE key IN ('hwm_balance', 'peak_balance_sniper', 'peak_balance_live')").fetchall()
                for r in rows:
                    try:
                        v = float(r[0])
                        if v > 0:
                            peaks.append(v)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            log.warning(f"[HWM-RESTORE] Error reading peak balance: {e}")
        highest = max(peaks)
        log.info(f"[HWM-RESTORE] Historical High-Water Mark restored: ${highest:,.2f}")
        return highest

    def get_drawdown(self) -> float:
        """Calculates current drawdown from peak balance."""
        if not self.sim or not hasattr(self.sim, "balance"): return 0.0
        bal_obj = self.sim.balance
        current = bal_obj.total if hasattr(bal_obj, "total") else float(bal_obj)
        if current > self.max_balance:
            self.max_balance = current
            try:
                import sqlite3
                from datetime import datetime, UTC
                with sqlite3.connect("trading_data.db", timeout=5.0) as conn:
                    conn.execute("INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES ('hwm_balance', ?, ?)", (str(current), datetime.now(UTC).isoformat()))
                    conn.commit()
            except Exception:
                pass
        
        if self.max_balance == 0: return 0.0
        dd = ((self.max_balance - current) / self.max_balance) * 100
        return max(0.0, dd)

    def update_metrics(self):
        """Syncs metrics with the exchange/sim balance."""
        if self.sim:
            dd = self.get_drawdown()
            if dd > self.max_drawdown:
                self.max_drawdown = dd
        self.last_update = datetime.now(UTC)

    def check_llm_recovery(self):
        """Decays LLM failure count over time to allow recovery."""
        if self.llm_failures > 0 and self.last_llm_failure:
            # Recover 1 failure every 1 hour of healthy operation
            hours_since = (datetime.now(UTC) - self.last_llm_failure).total_seconds() / 3600
            if hours_since > 1:
                recovery = int(hours_since)
                self.llm_failures = max(0, self.llm_failures - recovery)
                if self.llm_failures == 0:
                    self.is_observe_only = False

    def get_total_balance(self) -> float:
        """Returns the current total balance from the sim/exchange."""
        if self.sim and hasattr(self.sim, "balance"):
            bal_obj = self.sim.balance
            return bal_obj.total if hasattr(bal_obj, "total") else float(bal_obj)
        return self.start_balance

    def can_open_new_trade(self, symbol) -> bool:
        """Portfolio Guard: Check if we can open a new trade for a specific symbol."""
        # Check per-symbol entry limit (allows scale-in / multi-trade up to MAX_POSITIONS_PER_SYMBOL)
        max_per_sym = int(getattr(config, "MAX_POSITIONS_PER_SYMBOL", 3))
        allow_scale_in = getattr(config, "ALLOW_SAME_COIN_SCALE_IN", True)
        
        if self.in_position.get(symbol, False) and not allow_scale_in:
            return False
            
        # Count distinct unique coins currently active
        unique_active_coins = sum(1 for s in self.in_position if self.in_position[s])
        max_global_coins = int(getattr(config, "GLOBAL_MAX_DISTINCT_COINS", getattr(config, "MAX_CONCURRENT_TRADES", 4)))
        
        # If this is a new coin and we are already at the max distinct coin ceiling, block
        if not self.in_position.get(symbol, False) and unique_active_coins >= max_global_coins:
            return False
            
        if self.get_drawdown() >= config.GLOBAL_MAX_DD_LIMIT:
            if not config.IS_REPLAY:
                log.warning("GLOBAL RISK ALERT: Max Drawdown Limit Reached. Entry Blocked.")
            return False
            
        return True

    def record_trade(self, entry: TradeEntry):
        """Standardized trade recording."""
        # Log to CSV File
        try:
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    entry.timestamp.isoformat(), entry.symbol, entry.ai_action, entry.side,
                    entry.entry_price, entry.exit_price, entry.qty, entry.pnl, entry.balance,
                    entry.regime, entry.phase, entry.quality, entry.sl_mode, entry.tp_mode, entry.trail_mode,
                    entry.mfe, entry.ai_notes, entry.pos_id or "N/A"
                ])
        except Exception as e:
            log.error(f"[CSV ERROR] Failed to write trade to {self.csv_file}: {e}")

        # Log to database
        try:
            self.db.log_trade(
                entry.symbol, entry.side, entry.ai_action, entry.pnl, entry.ai_notes,
                entry_price=entry.entry_price, exit_price=entry.exit_price, qty=entry.qty,
                balance=entry.balance, regime=entry.regime, phase=entry.phase, mfe=entry.mfe
            )
        except Exception as e:
            log.error(f"[DATABASE ERROR] Failed to log trade: {e}")
        
        # Log to legacy journal
        self.journal.add(entry)
        
        # Update metrics
        if entry.pnl > 0:
            self.global_streak = max(1, self.global_streak + 1)
        elif entry.pnl < 0:
            self.global_streak = min(-1, self.global_streak - 1)
            
        # Frequency capping update
        now = entry.timestamp
        self.global_trade_timestamps.append(now)
        # Prune older than 24h
        cutoff = now - timedelta(hours=24)
        self.global_trade_timestamps = [t for t in self.global_trade_timestamps if t > cutoff]

        # FIX: Populate post_trade_notes for session_pnl tracking
        self.post_trade_notes.append({
            'timestamp': entry.timestamp,
            'symbol': entry.symbol,
            'pnl': entry.pnl,
            'side': entry.side
        })
        
        # Prune to last 50 trades to prevent memory bloat
        if len(self.post_trade_notes) > 50:
            self.post_trade_notes = self.post_trade_notes[-50:]

    def get_trades_last_24h(self) -> int:
        """Returns the number of trades taken in the rolling 24h window."""
        return len(self.global_trade_timestamps)

    def get_session_performance(self) -> float:
        """Returns total PnL from post_trade_notes."""
        return sum(t.get('pnl', 0.0) for t in self.post_trade_notes)

    def save_state(self, filepath="state.json"):
        """Persists minimal state to disk."""
        data = {
            "max_balance": self.max_balance,
            "max_drawdown": self.max_drawdown,
            "global_streak": self.global_streak,
            "llm_failures": self.llm_failures,
            "is_halted": self.is_halted
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load_state(self, filepath="state.json"):
        """Reloads persisted state."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    self.max_balance = data.get("max_balance", self.max_balance)
                    self.max_drawdown = data.get("max_drawdown", self.max_drawdown)
                    self.global_streak = data.get("global_streak", self.global_streak)
                    self.llm_failures = data.get("llm_failures", self.llm_failures)
                    self.is_halted = data.get("is_halted", self.is_halted)
            except Exception as e:
                log.warning(f"Failed to load state from {filepath}: {e}. Starting with default state.")
