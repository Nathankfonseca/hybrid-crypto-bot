import ccxt
import pandas as pd
import numpy as np
import ta

def fetch_macro_data(symbol='BTC/USDT', timeframe='1h', limit=1000):
    exchange = ccxt.bybit({'enableRateLimit': True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def apply_macro_features(df):
    # Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # ATR (14 and 50 periods)
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_50'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=50)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Distances to SMA (Z-score like approach using SMA as mean)
    sma_50 = ta.trend.sma_indicator(df['close'], window=50)
    sma_200 = ta.trend.sma_indicator(df['close'], window=200)
    
    df['dist_sma_50'] = (df['close'] - sma_50) / sma_50
    df['dist_sma_200'] = (df['close'] - sma_200) / sma_200
    
    df.dropna(inplace=True)
    return df

def generate_macro_labels(df, lookahead=24):
    # Future return over the next 24 hours (since it's a 1H chart)
    future_return = (df['close'].shift(-lookahead) - df['close']) / df['close']
    
    # 1: Buy (Acima da mediana), 0: Sell (Abaixo da mediana) para balanceamento 50/50
    median_ret = future_return.median()
    df['label'] = np.where(future_return > median_ret, 1, 0)
    
    df.dropna(inplace=True)
    return df
