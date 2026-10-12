import pandas as pd
from datetime import datetime, UTC
import os

files = [f for f in os.listdir("data/historical") if f.endswith(".csv")]
for f in files:
    path = os.path.join("data/historical", f)
    df = pd.read_csv(path)
    if not df.empty:
        start = datetime.fromtimestamp(df["timestamp"].iloc[0]/1000, UTC)
        end = datetime.fromtimestamp(df["timestamp"].iloc[-1]/1000, UTC)
        print(f"{f}: {start} to {end} ({len(df)} rows)")
    else:
        print(f"{f}: Empty")
