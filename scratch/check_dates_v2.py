import pandas as pd
from datetime import datetime, UTC
import os

data_dir = os.path.abspath("data/historical")
print(f"Checking directory: {data_dir}")
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
for f in files:
    path = os.path.join(data_dir, f)
    df = pd.read_csv(path)
    if not df.empty:
        start = datetime.fromtimestamp(df["timestamp"].iloc[0]/1000, UTC)
        end = datetime.fromtimestamp(df["timestamp"].iloc[-1]/1000, UTC)
        print(f"{f}: {start} to {end} ({len(df)} rows)")
    else:
        print(f"{f}: Empty")
