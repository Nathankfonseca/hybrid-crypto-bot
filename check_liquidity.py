import ccxt
import time
bybit = ccxt.bybit()
print("Checking tickers...")
for i in range(3):
    t_brl = bybit.fetch_ticker("BTC/BRL")
    print(f"[{i}] BTC/BRL Last: {t_brl['last']} | Bid: {t_brl.get('bid')} | Ask: {t_brl.get('ask')}")
    time.sleep(3)
