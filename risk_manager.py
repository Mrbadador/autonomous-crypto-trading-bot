# risk_manager.py — V22.0 Dynamic Risk Manager (Fractional Kelly Criterion)
#
# Replaces fixed position sizing with a mathematically grounded Kelly-based
# allocation that scales with the LLM Boardroom's conviction score. A hard
# percentage cap ensures no single trade ever risks more than the configured
# maximum, regardless of edge strength.

import logging
import socket
import aiohttp
import math
import dns_fix  # noqa: F401
import ccxt.async_support as ccxt
import config  # noqa: F401 — required for _ensure_corr_exchange()
from microstructure_metrics import (
    calculate_empirical_growth_optimal_kelly,
    calculate_bayesian_smoothed_kelly
)

log = logging.getLogger("risk_manager")


class RiskManager:
    """
    V22.0 Dynamic Position Sizer.

    Uses the Half-Kelly Criterion as a theoretically optimal base fraction,
    then scales it by the Boardroom's conviction score (0-100) so that
    high-confidence setups receive larger allocations and borderline setups
    receive almost nothing.

    Args:
        historical_win_rate: Empirical win rate from the trade ledger (e.g. 0.65).
        avg_risk_reward:     Average risk/reward ratio (e.g. 1.5 means 1.5R reward per 1R risk).
        max_risk_cap:        Hard upper limit on capital risked per trade (e.g. 0.05 = 5%).
    """

    def __init__(
        self,
        historical_win_rate: float = 0.65,
        avg_risk_reward: float = 1.5,
        max_risk_cap: float = None,
        max_positions: int = 5,       # 5-position Top 5 fleet limit
    ):
        self.win_rate = max(0.01, min(0.99, historical_win_rate))
        self.b = max(0.1, avg_risk_reward)   # Odds ratio (Reward / Risk)
        # Inherit institutional hard cap from config (default 0.5% = 0.005)
        if max_risk_cap is None:
            cfg_risk_pct = float(getattr(config, "RISK_PER_TRADE_PCT", 0.5))
            self.max_cap = cfg_risk_pct / 100.0 if cfg_risk_pct > 0.05 else cfg_risk_pct
        else:
            self.max_cap = max(0.001, min(0.10, max_risk_cap))
        self.max_positions = max_positions
        # Lazy-initialized — created inside the async event loop on first use.
        self._corr_exchange = None
        self._corr_session  = None
        log.info(
            f"[RISK-MANAGER] Online (Async). Base Kelly calculated on {self.win_rate * 100:.0f}% WR "
            f"| {self.b}R avg reward | {self.max_cap * 100:.0f}% hard cap."
        )

    async def _ensure_corr_exchange(self):
        """Lazy-initialize ccxt with a ThreadedResolver session (event-loop safe)."""
        if self._corr_session is None or self._corr_session.closed:
            connector = aiohttp.TCPConnector(
                resolver=aiohttp.ThreadedResolver(),
                family=socket.AF_INET,
                use_dns_cache=True,
                ttl_dns_cache=300,
            )
            self._corr_session = aiohttp.ClientSession(connector=connector)

        if self._corr_exchange is None:
            ex_id = getattr(config, "EXCHANGE_ID", "bitget").lower()
            ex_class = getattr(ccxt, ex_id, None) or ccxt.bitget
            self._corr_exchange = ex_class({
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap" if ex_id == "bitget" else "future",
                    "adjustForTimeDifference": True,
                },
                "aiohttp_session": self._corr_session,
            })

    async def close(self):
        """Closes the ccxt exchange and underlying aiohttp session."""
        if self._corr_exchange:
            await self._corr_exchange.close()
        if self._corr_session and not self._corr_session.closed:
            await self._corr_session.close()

    def _get_live_win_rate(self) -> tuple[float, int]:
        """[P1-FIX] Sync DB read with short retry.  Must NOT be called from the event loop
        directly — it uses blocking SQLite I/O.  It is only called from calculate_kelly_fraction()
        which itself is only called synchronously inside get_position_size() after we're already
        off the hot async path via asyncio.to_thread() in callers that need it."""
        import sqlite3
        import os
        try:
            import asyncio
            asyncio.get_running_loop()
            log.warning("[RISK-MANAGER] _get_live_win_rate called synchronously on active event loop. Use asyncio.to_thread.")
        except RuntimeError:
            pass

        for attempt in range(5):
            try:
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data.db")
                conn = sqlite3.connect(db_path, timeout=30.0)
                # HIGH-03: Cross-fleet win rate across Sniper, Berserker, and MARL
                query = """
                    SELECT pnl FROM trades WHERE status IN ('closed', 'CLOSED')
                    UNION ALL
                    SELECT pnl FROM berserker_trades WHERE status IN ('closed', 'CLOSED')
                    UNION ALL
                    SELECT pnl FROM marl_live_trades WHERE status IN ('closed', 'CLOSED')
                """
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                conn.close()
                total_trades = len(rows)
                if total_trades < 20:
                    return self.win_rate, total_trades
                wins = sum(1 for r in rows if float(r[0]) > 0)
                return float(wins) / total_trades, total_trades
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < 4:
                    import time as _t
                    # [P1-FIX] Short fixed sleep is safe here because this method should
                    # ONLY be called from asyncio.to_thread() or a non-async path.
                    # Do NOT call time.sleep() from within an async coroutine directly.
                    _t.sleep(0.1 * (2 ** attempt))  # 0.1, 0.2, 0.4, 0.8s — max 1.5s total
                else:
                    return self.win_rate, 0
            except Exception:
                return self.win_rate, 0
        return self.win_rate, 0

    def calculate_kelly_fraction(self) -> float:
        """
        Calculates the theoretical Kelly fraction and returns Half-Kelly for safety.

        Full Kelly: f* = W - (1-W) / B
        Half Kelly:  f  = f* / 2

        Returns:
            float: Half-Kelly fraction, or 0.0 if the edge is mathematically negative.
        """
        live_win_rate, count = self._get_live_win_rate()
        log.info(f"[RISK-MANAGER] Sample Size: {count} | Using Win-Rate: {live_win_rate:.2%}")

        q = 1.0 - live_win_rate
        full_kelly = live_win_rate - (q / self.b)

        if full_kelly <= 0:
            log.warning("[RISK-MANAGER] Full Kelly is negative — edge is unprofitable at current win rate / RR.")
            return 0.0

        return full_kelly / 2.0  # Half-Kelly for volatility dampening

    def _get_historical_returns(self) -> list[float]:
        """Fetches historical trade return fractions from trading_data.db."""
        import sqlite3
        import os
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data.db")
        returns = []
        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                st_rows = conn.execute("""
                    SELECT pnl, size, entry_price 
                    FROM swing_trades 
                    WHERE pnl IS NOT NULL
                    ORDER BY id DESC LIMIT 50
                """).fetchall()
                for r in st_rows:
                    pnl = float(r[0] or 0.0)
                    sz = float(r[1] or 0.0)
                    ep = float(r[2] or 0.0)
                    if sz > 0 and ep > 0:
                        margin_est = (sz * ep) / 5.0  # Normalized to 5x margin base
                        returns.append(pnl / margin_est if margin_est > 0 else 0.0)
                    elif pnl != 0.0:
                        returns.append(pnl / 2.0)  # Normalized fallback risk unit

                t_rows = conn.execute("""
                    SELECT pnl, qty, entry_price FROM trades 
                    WHERE status IN ('closed', 'CLOSED') AND pnl IS NOT NULL
                    ORDER BY id DESC LIMIT 50
                """).fetchall()
                for r in t_rows:
                    pnl = float(r[0] or 0.0)
                    q = float(r[1] or 0.0)
                    ep = float(r[2] or 0.0)
                    if q > 0 and ep > 0:
                        margin_est = (q * ep) / 5.0
                        returns.append(pnl / margin_est if margin_est > 0 else 0.0)
                    else:
                        returns.append(pnl / 1.50)
        except Exception as e:
            log.debug(f"[RISK-MANAGER] Return history read note: {e}")
        return returns

    def calculate_empirical_kelly_fraction(self) -> float:
        """
        Calculates distribution-free Quarter-Kelly fraction using empirical log-wealth optimization.
        Avoids Gaussian over-leveraging and accounts for actual skewness/kurtosis of trade returns.
        """
        returns = self._get_historical_returns()
        if len(returns) < 5:
            # Fallback to discrete half-kelly if trade history is fresh
            return self.calculate_kelly_fraction()

        optimal_f = calculate_bayesian_smoothed_kelly(
            returns=returns,
            fraction=0.25,  # Quarter-Kelly for maximum drawdown dampening
            max_cap=self.max_cap if self.max_cap > 0 else 0.25,
            min_cap=min(0.005, self.max_cap)
        )
        log.info(f"[RISK-MANAGER] Empirical Bayesian Quarter-Kelly Fraction: {optimal_f:.2%} (Sample: {len(returns)} trades)")
        return optimal_f

    async def _calculate_diversification_penalty(self, symbol: str, current_positions: list) -> float:
        """
        Queries Binance REST for the last 24h of 15m price data for the new symbol
        and currently open positions. Calculates Pearson correlation coefficient.
        If the average correlation is > 0.85, returns 0.5. Otherwise 1.0.
        """
        import pandas as pd
        import numpy as np

        if not current_positions or symbol is None:
            return 1.0

        # Ensure the exchange + session are ready (lazy async init)
        await self._ensure_corr_exchange()

        # P1-PATCH: Use the shared class-level exchange singleton.
        # Eliminates per-call connection leak and prevents repeated object creation.
        exchange = self._corr_exchange
        try:
            # Fetch 24h of 15m data (96 candles)
            limit = 96
            
            async def get_close_prices(sym):
                ohlcv = await exchange.fetch_ohlcv(sym, timeframe='15m', limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                return df['close'].astype(float)
            
            new_sym_closes = await get_close_prices(symbol)
            correlations = []
            
            for pos_sym in current_positions:
                pos_closes = await get_close_prices(pos_sym)
                # Align series to minimum length
                min_len = min(len(new_sym_closes), len(pos_closes))
                if min_len < 10:
                    continue
                corr = new_sym_closes.iloc[-min_len:].corr(pos_closes.iloc[-min_len:])
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                avg_corr = sum(correlations) / len(correlations)
                penalty_multiplier = 0.5 if avg_corr > 0.85 else 1.0
                log.info(f"[RISK-MANAGER] Correlation Audit: {avg_corr:.2f}. Diversification Penalty: {penalty_multiplier}x applied.")
                return penalty_multiplier

        except Exception as e:
            log.error(f"[RISK-MANAGER] Diversification Penalty failed: {e}")
        
        return 1.0
    def get_portfolio_var(self, current_balance: float, new_trade_risk: float = 0.0) -> bool:
        """
        V26.2 VaR Circuit Breaker.
        Sums the potential dollar risk of all open positions (qty * |entry - sl|)
        across BOTH standard and Berserker fleets.
        Returns False (VETO) if total potential loss > 10% of the account balance.
        """
        import sqlite3
        import os
        
        total_latent_var = 0.0
        
        try:
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_data.db")
            conn = sqlite3.connect(db_path, timeout=30.0)
            
            # Query all active fleet position tables including swing_open_positions
            rows = []
            for tbl in ["open_positions", "berserker_open_positions", "marl_live_open_positions", "mean_reversion_open_positions"]:
                try:
                    cursor = conn.execute(f"SELECT qty, entry_price, stop_loss FROM {tbl}")
                    rows.extend(cursor.fetchall())
                except Exception:
                    pass
            try:
                cursor = conn.execute("SELECT size AS qty, entry_price, stop_loss FROM swing_open_positions WHERE status = 'ACTIVE' OR status IS NULL")
                rows.extend(cursor.fetchall())
            except Exception:
                pass
            conn.close()
            
            import math
            for qty, entry, sl in rows:
                if qty and entry:
                    try:
                        f_qty = float(qty)
                        f_entry = float(entry)
                        if not (math.isfinite(f_qty) and math.isfinite(f_entry)):
                            continue
                        if sl is not None:
                            f_sl = float(sl)
                            if math.isfinite(f_sl):
                                pos_risk = f_qty * abs(f_entry - f_sl)
                            else:
                                pos_risk = f_qty * f_entry * 0.05
                        else:
                            # Positions without explicit SL (e.g. Mean Reversion) conservatively assigned 5% latent notional risk
                            pos_risk = f_qty * f_entry * 0.05
                        total_latent_var += pos_risk
                    except (TypeError, ValueError) as conv_err:
                        log.warning(f"[VaR-AUDIT] Could not convert position row to float: {conv_err}")
                    
        except Exception as e:
            log.error(f"[VaR-AUDIT] Database read error during risk aggregation: {e}")
            pass
            
        total_var = total_latent_var + new_trade_risk
        
        # HIGH-02: Fallback to system_state or INITIAL_BALANCE if current_balance <= 0
        if current_balance <= 0:
            try:
                with sqlite3.connect(db_path, timeout=5.0) as b_conn:
                    b_cur = b_conn.execute("SELECT value FROM system_state WHERE key IN ('balance_sniper', 'balance_marl_live', 'account_balance') ORDER BY updated_at DESC LIMIT 1")
                    b_row = b_cur.fetchone()
                    if b_row and float(b_row[0]) > 0:
                        current_balance = float(b_row[0])
            except Exception:
                pass
        if current_balance <= 0:
            import config as _cfg
            current_balance = float(getattr(_cfg, "INITIAL_BALANCE", 58.77))

        var_pct = (total_var / current_balance) if current_balance > 0 else 0.0
        
        log.info(f"[VaR-AUDIT] Portfolio Risk: ${total_var:.2f} ({var_pct:.2%}) | Limit: 10.00%")
        
        if var_pct > 0.10:
            log.critical(f"[VaR-VETO] PORTFOLIO HEAT CRITICAL! ${total_var:.2f} risks {var_pct:.2%}. Vetoing trade.")
            return False
            
        return True

    async def get_position_size(
        self, account_balance: float, conviction_score: float,
        symbol: str = None, current_positions: list = None
    ) -> tuple[float, float]:
        """
        Scales the Half-Kelly by the AI Conviction Score and applies the hard cap.

        Args:
            account_balance:  Current usable balance in USD.
            conviction_score: Boardroom conviction score, range 0-100.
            symbol:           Symbol being evaluated for entry.
            current_positions:List of symbols currently open in the portfolio.

        Returns:
            Tuple of (allocated_capital_usd, final_risk_pct).
            Returns (0.0, 0.0) if the mathematical edge is negative.
        """
        if account_balance <= 0.0 or not math.isfinite(account_balance):
            log.warning("[RISK-MANAGER] Invalid or non-positive account balance. Returning zero allocation.")
            return 0.0, 0.0

        import asyncio
        base_risk_pct = await asyncio.to_thread(self.calculate_empirical_kelly_fraction)

        if base_risk_pct <= 0.0:
            log.warning("[RISK-MANAGER] Edge is negative or zero. Returning zero allocation — VETO trade.")
            return 0.0, 0.0

        # Scale risk linearly by conviction: normalize whether passed as [0.0, 1.0] or [0.0, 100.0]
        c_score = float(conviction_score) if conviction_score is not None else 0.75
        if c_score <= 1.0 and c_score > 0.0:
            conviction_multiplier = max(0.0, min(1.0, c_score))
        else:
            conviction_multiplier = max(0.0, min(100.0, c_score)) / 100.0
        adjusted_risk_pct = base_risk_pct * conviction_multiplier

        # Apply Block 3 Diversification Penalty
        if symbol and current_positions:
            penalty = await self._calculate_diversification_penalty(symbol, current_positions)
            adjusted_risk_pct *= penalty

        # Apply Asset Tier Sizing Multipliers (BTC/ETH 1.5x, SOL/BNB 0.8x, Alts 0.5x)
        if symbol:
            tier_multiplier = getattr(config, "get_asset_tier_multiplier", lambda s: 1.0)(symbol)
            adjusted_risk_pct *= tier_multiplier

        # Apply hard safety cap regardless of Kelly output
        final_risk_pct = min(adjusted_risk_pct, self.max_cap)
        allocated_capital = account_balance * final_risk_pct

        log.info(
            f"[RISK-MANAGER] Conviction: {conviction_score:.0f}/100 | "
            f"Half-Kelly: {base_risk_pct * 100:.2f}% → Adjusted: {adjusted_risk_pct * 100:.2f}% → "
            f"Final (capped): {final_risk_pct * 100:.2f}% | "
            f"Allocating: ${allocated_capital:.2f} of ${account_balance:.2f}"
        )
        return allocated_capital, final_risk_pct


# ---------------------------------------------------------------------------
# Standalone diagnostic (python risk_manager.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    balance = float(sys.argv[1]) if len(sys.argv) > 1 else 10_000.0
    conviction = float(sys.argv[2]) if len(sys.argv) > 2 else 92.0

    rm = RiskManager()

    async def main():
        try:
            size, pct = await rm.get_position_size(balance, conviction)
            print(f"\n{'='*60}")
            print(f"  Balance      : ${balance:,.2f}")
            print(f"  Conviction   : {conviction:.0f}/100")
            print(f"  Allocation   : ${size:,.2f}  ({pct * 100:.2f}% of balance)")
            print(f"{'='*60}\n")
        finally:
            await rm.close()

    asyncio.run(main())
