# vectorized_backtester.py — V17.0 High-Performance Time Machine
import pandas as pd
import numpy as np
import time
import sys

def generate_mock_data(rows=1_000_000):
    """Creates synthetic 5-minute OHLCV data for rapid high-scale testing."""
    print(f"[*] Synchronizing Core: Synthesizing {rows:,} market data rows...")
    start_t = time.time()
    
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=rows, freq="5min")
    
    # Simulating daily price diffusion
    daily_drift = 0.00001  # slight upward bias
    volatility = 0.0005
    returns = np.random.normal(loc=daily_drift, scale=volatility, size=rows)
    
    # Compute Price Series
    price_init = 50000.0
    close = price_init * np.exp(np.cumsum(returns))
    
    # Generate OHLCV context around close
    noise = np.random.uniform(0.999, 1.001, rows)
    open_p = close * noise
    high = np.maximum(close, open_p) * np.random.uniform(1.0001, 1.0015, size=rows)
    low = np.minimum(close, open_p) * np.random.uniform(0.9985, 0.9999, size=rows)
    volume = np.random.uniform(1, 100, size=rows)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    
    elapsed = time.time() - start_t
    print(f"[+] Dataset generated successfully in {elapsed:.2f} seconds.")
    return df

def run_vectorized_backtest(df):
    """Executes backtest loop executing 1M candle iterations instantly via Pandas Vectorization."""
    print("[*] Engaging Vector Engine...")
    t_start = time.time()
    
    # ==========================================
    # STEP 1: V17.0 ORACLE CALCULATIONS
    # ==========================================
    
    # A. Vectorized VWAP Engine
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_vol = df["volume"].cumsum()
    df["vwap"] = (df["typical_price"] * df["volume"]).cumsum() / cum_vol.replace(0, 1)
    
    # B. Vectorized RSI-14 Engine
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # Clean up artifacts safely
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # ==========================================
    # STEP 2: BERSERKER SIGNAL LOGIC
    # ==========================================
    
    # V17.0 Entry: Price drops below RSI limit while still above support VWAP (Pullback entry)
    # Example threshold: Oversold bounce (rsi < 30) above VWAP foundation.
    df["signal"] = (df["close"] > df["vwap"]) & (df["rsi_14"] < 30)
    
    # Lag signal by 1 cycle to model realistic execution on the NEXT bar's Open.
    # Prevents future-leak lookahead bias.
    df["position"] = df["signal"].shift(1).fillna(0).astype(int)
    
    # ==========================================
    # STEP 3: STRATEGY PERFORMANCE MATRIX
    # ==========================================
    
    # Raw Asset performance curve
    df["market_pct"] = df["close"].pct_change().fillna(0)
    
    # Scaled Strategy Return
    df["strategy_pct"] = df["position"] * df["market_pct"]
    
    # Compounded Equity Curve Generation
    df["equity_curve"] = (1 + df["strategy_pct"]).cumprod()
    
    # ==========================================
    # STEP 4: INSTITUTIONAL RISK METRICS
    # ==========================================
    
    # Max Drawdown Calculus
    running_peak = df["equity_curve"].cummax()
    drawdowns = (df["equity_curve"] - running_peak) / running_peak.replace(0, 1)
    max_dd = abs(drawdowns.min())
    
    # Annualized Vectorized Sharpe Ratio
    # Standard scaling assumes 105,120 intervals (5m) in 1 active trading year
    trading_windows_per_year = 105120
    mean_ret = df["strategy_pct"].mean()
    std_dev = df["strategy_pct"].std()
    
    sharpe_raw = (mean_ret / std_dev) if std_dev > 0 else 0.0
    sharpe_annualized = sharpe_raw * np.sqrt(trading_windows_per_year)
    
    total_ret = df["equity_curve"].iloc[-1] - 1.0
    
    exec_t = time.time() - t_start
    
    # ==========================================
    # FINAL REPORT CONSOLE OUTPUT
    # ==========================================
    
    print("\n" + "=" * 60)
    print("         THE V17.0 'TIME MACHINE' BACKTEST COMPLETED")
    print("=" * 60)
    print(f"  Candles Processed   : {len(df):,}")
    print(f"  Computation Time    : {exec_t:.4f} seconds (Instant Vector Mode)")
    print("-" * 60)
    print(f"  Final Equity Return : {total_ret:+.2%}")
    print(f"  Maximum Drawdown    : {max_dd:.2%}")
    print(f"  Sharpe Ratio (Ann.) : {sharpe_annualized:.4f}")
    print("=" * 60)
    
    # Quick Win/Loss Audit
    total_trades = df["position"].sum()
    print(f"\n  [Audit] Active Bars in Strategy: {total_trades:,} cycles")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        # Bootstrap Data
        market_data = generate_mock_data(rows=1_000_000)
        
        # Fire Vector Engine
        run_vectorized_backtest(market_data)
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] Engine Failure: {e}")
        sys.exit(1)
