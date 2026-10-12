import logging
import asyncio
import config

log = logging.getLogger(__name__)

class NotificationManager:
    def __init__(self):
        self.enabled = False # Default to disabled until configured
        self.queue = asyncio.Queue()
        self._loop_task = None

    def start(self):
        """Starts the background worker for notifications."""
        if not self._loop_task:
            self._loop_task = asyncio.create_task(self._worker())
        log.info("NotificationManager worker started.")

    async def _worker(self):
        while True:
            msg = await self.queue.get()
            try:
                await self._send_to_telegram(msg)
            except Exception as e:
                log.error(f"Failed to send notification: {e}")
            finally:
                self.queue.task_done()

    def notify_trade(self, symbol, action, side, price, pnl=None):
        """Queues a trade notification."""
        emoji = "🚀" if "open" in action or "add" in action else "🎯"
        if pnl is not None:
            emoji = "💰" if pnl > 0 else "📉"
            
        msg = f"{emoji} [{symbol}] {action.upper()} {side.upper()} @ {price:.2f}"
        if pnl is not None:
            msg += f" | PnL: {pnl:.2f}"
            
        self.queue.put_nowait(msg)

    def notify_alert(self, level, message):
        """Queues a system alert notification."""
        emoji = "⚠️" if level == "warning" else "🚨"
        msg = f"{emoji} SYSTEM ALERT: {message}"
        self.queue.put_nowait(msg)

    def stop(self):
        """Stops the background worker."""
        if self._loop_task:
            self._loop_task.cancel()
            log.info("NotificationManager worker stopped.")
            self._loop_task = None

    async def _send_to_telegram(self, message):
        """Placeholder for real Telegram API call."""
        log.info(f"NOTIFICATION STUB: {message}")
        await asyncio.sleep(0.1) # Simulate network IO
