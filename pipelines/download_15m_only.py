import ccxt
import pandas as pd
import time
import os
from datetime import datetime

def download_historical_data(symbol='BTC/USDT'):
    exchange = ccxt.bybit({'enableRateLimit': True})
    os.makedirs('data', exist_ok=True)
    
    now = exchange.milliseconds()
    
    # --- 15M DATA (ALL AVAILABLE) ---
    print(f"\n[{datetime.now()}] Iniciando download de TODOS os dados disponíveis de 15 Minutos...")
    since_15m = exchange.parse8601('2018-01-01T00:00:00Z')
    current_since = since_15m
    
    all_15m_candles = []
    
    while current_since < now:
        try:
            candles = exchange.fetch_ohlcv(symbol, '15m', since=current_since, limit=1000)
            if not candles:
                break
            all_15m_candles.extend(candles)
            current_since = candles[-1][0] + 15 * 60 * 1000 # add 15 minutes
            
            if len(all_15m_candles) % 10000 == 0:
                print(f"Baixadas {len(all_15m_candles)} velas 15M... Ultima data: {pd.to_datetime(candles[-1][0], unit='ms')}")
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}. Tentando novamente...")
            time.sleep(2)
            
    df_15m = pd.DataFrame(all_15m_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_15m.drop_duplicates(subset=['timestamp'], inplace=True)
    df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms')
    df_15m.to_parquet('data/micro_15m.parquet', engine='pyarrow')
    print(f"[OK] Concluido! {len(df_15m)} velas de 15M salvas em data/micro_15m.parquet")

if __name__ == "__main__":
    download_historical_data()
