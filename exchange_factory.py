"""
Sovereign Swarm — Unified Exchange Factory
Dynamically instantiates sync and async CCXT exchanges for Binance or Bitget
based on config.EXCHANGE_ID and active credentials.
"""
import config
import logging
import aiohttp

log = logging.getLogger("exchange_factory")


def get_exchange_config(async_mode=False):
    """Builds standard exchange parameters for CCXT."""
    ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
    
    cfg = {
        "apiKey": getattr(config, "ACTIVE_API_KEY", ""),
        "secret": getattr(config, "ACTIVE_SECRET_KEY", ""),
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {
            "adjustForTimeDifference": True,
        }
    }

    if ex_id == "bitget":
        cfg["password"] = getattr(config, "ACTIVE_PASSPHRASE", "")
        cfg["options"]["defaultType"] = "swap"  # USDT-M Perpetual Futures
    elif ex_id in ["binance", "binanceusdm"]:
        cfg["options"]["defaultType"] = "future"  # USDⓈ-M Futures

    if async_mode:
        cfg["aiohttp_config"] = {
            "resolver": aiohttp.ThreadedResolver()
        }

    return ex_id, cfg


def create_sync_exchange():
    """Returns a synchronous CCXT exchange instance."""
    import ccxt
    ex_id, cfg = get_exchange_config(async_mode=False)
    exchange_class = getattr(ccxt, ex_id, None)
    if not exchange_class:
        exchange_class = ccxt.bitget
    return exchange_class(cfg)


def create_async_exchange():
    """Returns an asynchronous CCXT exchange instance."""
    import ccxt.async_support as ccxt_async
    ex_id, cfg = get_exchange_config(async_mode=True)
    exchange_class = getattr(ccxt_async, ex_id, None)
    if not exchange_class:
        exchange_class = ccxt_async.bitget
    return exchange_class(cfg)


def get_bitget_uta_client():
    """Returns the dedicated Bitget UTA V3 client for live Unified Account execution."""
    from bitget_uta_client import BitgetUTAClient
    return BitgetUTAClient()

