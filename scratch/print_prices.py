import json

with open('microstructure_state.json') as f:
    data = json.load(f)

syms = data.get('symbols', {})
for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'NEARUSDT', 'ADAUSDT', 'BNBUSDT', 'AVAXUSDT']:
    if s in syms:
        b = syms[s].get('best_bid')
        ofi = syms[s].get('normalized_ofi', 0)
        h = syms[s].get('hawkes_spoof_index', 0)
        print(f"{s:10} | Price: ${b:<10} | OFI: {ofi:+.4f} | Hawkes Spoof: {h:.3f}")
