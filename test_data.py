import ccxt
import pandas as pd
import numpy as np
import ta
import time

def build_volume_bars(df_1m, target_volume=5.0):
    vol_bars = []
    current_vol = 0
    if len(df_1m) == 0: return pd.DataFrame()
        
    bar_open = df_1m['open'].iloc[0]
    bar_high = df_1m['high'].iloc[0]
    bar_low = df_1m['low'].iloc[0]
    
    for idx, row in df_1m.iterrows():
        current_vol += row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        if current_vol >= target_volume:
            vol_bars.append({
                'timestamp': row['timestamp'],
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': row['close'],
                'volume': current_vol
            })
            current_vol = 0
            bar_open = row['close']
            bar_high = row['close']
            bar_low = row['close']
            
    df_vol = pd.DataFrame(vol_bars)
    return df_vol

bybit = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

try:
    print("Fetching 1h...")
    ohlcv_1h = bybit.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    print(f"Fetched {len(df_1h)} 1h candles.")
    
    print("Fetching 1m...")
    ohlcv_1m = bybit.fetch_ohlcv('BTC/USDT', '1m', limit=1000)
    df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    print(f"Fetched {len(df_1m)} 1m candles.")
    
    # Engenharia Macro
    df_1h['log_return'] = np.log(df_1h['close'] / df_1h['close'].shift(1))
    df_1h['atr_14'] = ta.volatility.average_true_range(df_1h['high'], df_1h['low'], df_1h['close'], window=14)
    df_1h['atr_50'] = ta.volatility.average_true_range(df_1h['high'], df_1h['low'], df_1h['close'], window=50)
    macd = ta.trend.MACD(df_1h['close'])
    df_1h['macd'] = macd.macd()
    df_1h['macd_signal'] = macd.macd_signal()
    sma_50 = ta.trend.sma_indicator(df_1h['close'], window=50)
    sma_200 = ta.trend.sma_indicator(df_1h['close'], window=200)
    df_1h['dist_sma_50'] = (df_1h['close'] - sma_50) / sma_50
    df_1h['dist_sma_200'] = (df_1h['close'] - sma_200) / sma_200
    df_1h.dropna(inplace=True)
    print(f"Len df_1h after dropna: {len(df_1h)}")
    
    # Engenharia Micro
    df_vol = build_volume_bars(df_1m, target_volume=5.0)
    df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
    df_vol['momentum'] = df_vol['close'] - df_vol['close'].shift(1)
    df_vol.dropna(inplace=True)
    print(f"Len df_vol after dropna: {len(df_vol)}")
    
    if len(df_1h) >= 96 and len(df_vol) >= 10:
        print("Ready for inference!")
    else:
        print("NOT READY FOR INFERENCE.")

except Exception as e:
    print("ERROR:", e)
