# core/ai_trader_bridge.py
import asyncio
import logging
import sys
import os

# Set up path to import db_manager from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from db_manager import DatabaseManager
except Exception:
    DatabaseManager = None

log = logging.getLogger("ai_trader_bridge")

class AITraderBridge:
    _session = None

    def __init__(self, api_key: str, endpoint=None):
        self.api_key = api_key
        # Severed the external bridge to api.ai4trade.ai
        self.endpoint = None
        self.headers = {}
        
        try:
            if DatabaseManager:
                self.db = DatabaseManager()
            else:
                self.db = None
        except Exception:
            self.db = None

    async def _get_session(self):
        """Stubbed out to prevent external network resource allocation."""
        return None

    async def broadcast_trade(self, symbol: str, side: str, price: float, size: float, confidence: float):
        """
        Saves a trade signal locally to our WAL trading_data.db via Database Manager.
        All external HTTP network dependencies are severed.
        """
        try:
            if self.db:
                # Save locally off-thread to prevent event loop blocking
                await asyncio.to_thread(
                    self.db.log_tradesync,
                    symbol=symbol,
                    side=side,
                    price=price,
                    quantity=size,
                    confidence=confidence,
                    source="SovereignSwarm_V30.1"
                )
                log.info(f"[LOCAL-SYNC] TradeSync telemetry saved locally for {symbol} ({side.upper()})")
                return True
        except Exception:
            pass  # Silent Fail per requirements
        return False

    @classmethod
    async def close(cls):
        """Stubbed out cleanly."""
        pass
