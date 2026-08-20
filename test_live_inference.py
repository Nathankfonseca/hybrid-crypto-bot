import ccxt
import time
import pandas as pd
import numpy as np
import torch
import joblib
import ta
from models.autoformer import Autoformer
from models.cnn_transformer import CNNTransformer

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
            
    return pd.DataFrame(vol_bars)

def test():
    bybit = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    device = torch.device('cpu')
    
    macro_scaler = joblib.load('models/macro_scaler.pkl')
    micro_scaler = joblib.load('models/micro_scaler.pkl')
    
    macro_features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
    micro_features = ['log_return', 'momentum']
    
    macro_model = Autoformer(num_features=len(macro_features), num_classes=2).to(device)
    macro_model.load_state_dict(torch.load('models/macro_autoformer.pt', map_location=device))
    macro_model.eval()
    
    micro_model = CNNTransformer(num_features=len(micro_features), num_classes=2).to(device)
    micro_model.load_state_dict(torch.load('models/micro_cnn.pt', map_location=device))
    micro_model.eval()

    for i in range(2):
        print(f"Fetch {i+1}...")
        ohlcv_1h = bybit.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
        df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        ohlcv_1m = bybit.fetch_ohlcv('BTC/USDT', '1m', limit=1000)
        df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
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
        
        df_vol = build_volume_bars(df_1m, target_volume=5.0)
        df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
        df_vol['momentum'] = df_vol['close'] - df_vol['close'].shift(1)
        df_vol.dropna(inplace=True)
        
        macro_input = df_1h.tail(96).copy()
        macro_input[macro_features] = macro_scaler.transform(macro_input[macro_features])
        x_macro = torch.tensor(macro_input[macro_features].values, dtype=torch.float32).unsqueeze(0).to(device)
        
        micro_input = df_vol.tail(10).copy()
        micro_input[micro_features] = micro_scaler.transform(micro_input[micro_features])
        x_micro = torch.tensor(micro_input[micro_features].values, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            macro_out = macro_model(x_macro)
            macro_probs = torch.softmax(macro_out, dim=1).squeeze().numpy()
            micro_out = micro_model(x_micro)
            micro_probs = torch.softmax(micro_out, dim=1).squeeze().numpy()
            
        print(f"Macro: BUY {macro_probs[1]*100:.2f}% | SELL {macro_probs[0]*100:.2f}%")
        print(f"Micro: BUY {micro_probs[1]*100:.2f}% | SELL {micro_probs[0]*100:.2f}%")
        
        print("Micro Tail(3) log_return:", micro_input['log_return'].tail(3).values)
        if i == 0:
            time.sleep(5)

if __name__ == '__main__':
    test()
