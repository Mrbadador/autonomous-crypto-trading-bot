# dns_fix.py — Environment DNS Patch
#
# aiodns/c-ares cannot contact DNS servers in this environment, but the OS
# socket resolver (getaddrinfo) works correctly.  This module monkey-patches
# ccxt's Exchange.open() method to inject aiohttp.ThreadedResolver into the
# TCPConnector that ccxt creates — replacing the hardcoded no-resolver call.
#
# Usage: import dns_fix  (at the top of any file that uses ccxt)
#        A single import patches ALL ccxt exchange instances globally.

import socket
import aiohttp
import asyncio
import sys
from ccxt.async_support.base.exchange import Exchange as _Exchange  # correct path

_original_open = _Exchange.open


def _patched_open(self):
    """Patched Exchange.open() that injects ThreadedResolver into ccxt's
    TCPConnector, bypassing the broken aiodns/c-ares resolver.

    Only takes effect when ccxt would create its own session (own_session=True
    and session is None).  Pre-supplied sessions are left untouched.
    """
    # --- Resolve asyncio_loop (same logic as original) ---
    if self.asyncio_loop is None:
        if sys.version_info >= (3, 7):
            self.asyncio_loop = asyncio.get_running_loop()
        else:
            self.asyncio_loop = asyncio.get_event_loop()
        self.throttler.loop = self.asyncio_loop

    # --- SSL context (same logic as original) ---
    if self.ssl_context is None:
        import ssl as _ssl
        self.ssl_context = (
            _ssl.create_default_context(cafile=self.cafile) if self.verify else self.verify
        )
        if self.ssl_context and self.safe_bool(self.options, 'include_OS_certificates', False):
            os_default_paths = _ssl.get_default_verify_paths()
            if os_default_paths.cafile and os_default_paths.cafile != self.cafile:
                self.ssl_context.load_verify_locations(cafile=os_default_paths.cafile)

    # --- Session/connector (PATCHED: inject ThreadedResolver) ---
    if self.own_session and self.session is None:
        self.tcp_connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            resolver=aiohttp.ThreadedResolver(),   # ← DNS FIX: bypass aiodns/c-ares
            family=socket.AF_INET,                 # ← force IPv4 for consistency
            enable_cleanup_closed=True,
        )
        self.session = aiohttp.ClientSession(
            connector=self.tcp_connector,
            trust_env=self.aiohttp_trust_env,
        )


_Exchange.open = _patched_open

import logging as _logging
_logging.getLogger("dns_fix").info(
    "[DNS-FIX] ccxt Exchange.open() patched — ThreadedResolver active for ALL ccxt instances."
)
