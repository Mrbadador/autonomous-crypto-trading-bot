import asyncio
import json
import sqlite3
import os
from basis_trader import BTCBasisTrader

async def run_verification():
    print("==================================================")
    print("   BTC CASH-AND-CARRY BASIS TRADER VERIFICATION")
    print("==================================================")
    
    # 1. Initialize
    trader = BTCBasisTrader()
    await trader.initialize_clients()
    print("[OK] 1. Engine initialized & DB tables verified.")
    
    # 2. Market State
    market = await trader.fetch_market_state()
    assert market is not None, "Failed to fetch live market state"
    print(f"[OK] 2. Live Market Data Ingested:")
    print(f"      Spot Price:        ${market['spot_price']:,.2f}")
    print(f"      Perp Price:        ${market['perp_price']:,.2f}")
    print(f"      Basis Spread:      ${market['basis_spread']:+.2f} ({market['basis_dislocation']*100:+.3f}%)")
    print(f"      8h Funding Rate:   {market['funding_rate_8h']*100:+.4f}%")
    print(f"      Annualized Yield:  {market['annualized_yield']*100:+.2f}%")
    print(f"      Hurdle Threshold:  {trader.min_annual_yield*100:.2f}%")
    
    # 3. Telemetry Emission
    trader.publish_telemetry(market)
    await asyncio.sleep(0.5)
    
    conn = sqlite3.connect("trading_data.db", timeout=5.0)
    row = conn.execute("SELECT value FROM system_state WHERE key = 'basis_trader_telemetry'").fetchone()
    conn.close()
    assert row is not None, "Telemetry not found in system_state"
    telem = json.loads(row[0])
    print("[OK] 3. Real-Time Telemetry Heartbeat Verified in system_state:")
    print(f"      Status:            {telem.get('status')}")
    print(f"      Annualized Yield:  {telem.get('annualized_yield_pct')}%")
    print(f"      Net Delta:         ${telem.get('net_delta_usd'):.2f}")
    print(f"      Capital Slice:     ${telem.get('capital_allocated_usd'):,.2f}")
    
    # 4. Clean shutdown
    if trader.exchange:
        await trader.exchange.close()
    if trader.uta_client:
        await trader.uta_client.close()
    print("[OK] 4. Clean shutdown completed. All tests PASSED!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_verification())
