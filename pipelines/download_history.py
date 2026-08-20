import ccxt
import pandas as pd
import time
import os
import sys
from datetime import datetime

def download_historical_data(symbol='BTC/USDT'):
    exchange = ccxt.bybit({'enableRateLimit': True})
    os.makedirs('data', exist_ok=True)
    
    file_suffix = symbol.replace('/', '')
    
    # --- 1H DATA (ALL AVAILABLE) ---
    print(f"[{datetime.now()}] Iniciando download de TODOS os dados disponíveis de 1 Hora para {symbol}...")
    since_1h = exchange.parse8601('2018-01-01T00:00:00Z') # Bybit spot launched later, but we use a safe early date
    now = exchange.milliseconds()
    
    all_1h_candles = []
    current_since = since_1h
    
    while current_since < now:
        try:
            candles = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
            if not candles:
                break
            all_1h_candles.extend(candles)
            current_since = candles[-1][0] + 60 * 60 * 1000 # add 1 hour
            print(f"Baixadas {len(all_1h_candles)} velas 1H... Última data: {pd.to_datetime(candles[-1][0], unit='ms')}")
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}. Tentando novamente...")
            time.sleep(2)
            
    df_1h = pd.DataFrame(all_1h_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1h.drop_duplicates(subset=['timestamp'], inplace=True)
    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms')
    df_1h.to_parquet(f'data/macro_1h_{file_suffix}.parquet', engine='pyarrow')
    print(f"[OK] Concluido! {len(df_1h)} velas de 1H salvas em data/macro_1h_{file_suffix}.parquet")

    # --- 15M DATA (ALL AVAILABLE) ---
    print(f"\n[{datetime.now()}] Iniciando download de TODOS os dados disponíveis de 15 Minutos para {symbol}...")
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
    df_15m.to_parquet(f'data/micro_15m_{file_suffix}.parquet', engine='pyarrow')
    print(f"[OK] Concluido! {len(df_15m)} velas de 15M salvas em data/micro_15m_{file_suffix}.parquet")

    # --- 1M DATA (LAST 2 YEARS) ---
    print(f"\n[{datetime.now()}] Iniciando download dos ultimos 2 Anos de dados de 1 Minuto para {symbol}...")
    since_1m = now - (2 * 365 * 24 * 60 * 60 * 1000)
    current_since = since_1m
    
    all_1m_candles = []
    
    while current_since < now:
        try:
            candles = exchange.fetch_ohlcv(symbol, '1m', since=current_since, limit=1000)
            if not candles:
                break
            all_1m_candles.extend(candles)
            current_since = candles[-1][0] + 60 * 1000 # add 1 minute
            
            if len(all_1m_candles) % 50000 == 0:
                print(f"Baixadas {len(all_1m_candles)} velas 1M... Ultima data: {pd.to_datetime(candles[-1][0], unit='ms')}")
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Erro: {e}. Tentando novamente...")
            time.sleep(2)
            
    df_1m = pd.DataFrame(all_1m_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_1m.drop_duplicates(subset=['timestamp'], inplace=True)
    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
    df_1m.to_parquet(f'data/micro_1m_{file_suffix}.parquet', engine='pyarrow')
    print(f"[OK] Concluido! {len(df_1m)} velas de 1M salvas em data/micro_1m_{file_suffix}.parquet")
    print("\nDownload Completo! Pode parar por aqui.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_historical_data(sys.argv[1])
    else:
        download_historical_data()
