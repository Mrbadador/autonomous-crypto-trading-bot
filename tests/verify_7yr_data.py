import pandas as pd
import json

files = {
    "BTC/USDT": "historical_btc_usdt.csv",
    "ETH/USDT": "historical_eth_usdt.csv",
    "SOL/USDT": "historical_sol_usdt.csv",
    "XRP/USDT": "historical_xrp_usdt.csv"
}

print("\n" + "="*70)
print("   DATASET & LOSS HARVEST VERIFICATION CHECK")
print("="*70)

total_candles = 0
for sym, f in files.items():
    df = pd.read_csv(f)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    start_dt = df["timestamp"].iloc[0]
    end_dt = df["timestamp"].iloc[-1]
    days = (end_dt - start_dt).days
    years = days / 365.25
    total_candles += len(df)
    print(f"  {sym:<10}: {len(df):>6,} candles | {start_dt.strftime('%Y-%m-%d')} -> {end_dt.strftime('%Y-%m-%d')} | {years:.1f} years ({days:,} days)")

print("-" * 70)
print(f"  TOTAL CANDLES PROCESSED: {total_candles:,}")

with open("loss_harvest_2019_2026.json", "r") as f:
    losses = json.load(f)

print(f"  TOTAL LOSS RECORDS EXTRACTED FOR AI: {len(losses):,} trades")
print(f"  FIRST HARVESTED LOSS DATE: {losses[0]['entry_time']}")
print(f"  LAST HARVESTED LOSS DATE : {losses[-1]['entry_time']}")
print("="*70 + "\n")
