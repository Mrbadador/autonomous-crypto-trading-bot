import ccxt

ex = ccxt.bitget()
syms = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'ADA/USDT', 'XRP/USDT', 'AVAX/USDT', 'NEAR/USDT', 'DOGE/USDT']
for s in syms:
    try:
        t = ex.fetch_ticker(s)
        print(f"{s:10} | Price: {t['last']:<10} | 24h Change: {t.get('percentage', 0):+.2f}%")
    except Exception as e:
        print(s, e)
