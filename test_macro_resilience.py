import ccxt
import pandas as pd
import numpy as np
import torch
import joblib
import ta
from models.autoformer import Autoformer

def test():
    bybit = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    device = torch.device('cpu')
    
    macro_scaler = joblib.load('models/macro_scaler.pkl')
    macro_features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
    
    macro_model = Autoformer(num_features=len(macro_features), num_classes=2).to(device)
    macro_model.load_state_dict(torch.load('models/macro_autoformer.pt', map_location=device))
    macro_model.eval()
    
    ohlcv_1h = bybit.fetch_ohlcv('BTC/USDT', '1h', limit=1000)
    df_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    print("Baseline:")
    baseline_df = df_1h.copy()
    baseline_df['log_return'] = np.log(baseline_df['close'] / baseline_df['close'].shift(1))
    baseline_df['atr_14'] = ta.volatility.average_true_range(baseline_df['high'], baseline_df['low'], baseline_df['close'], window=14)
    baseline_df['atr_50'] = ta.volatility.average_true_range(baseline_df['high'], baseline_df['low'], baseline_df['close'], window=50)
    macd = ta.trend.MACD(baseline_df['close'])
    baseline_df['macd'] = macd.macd()
    baseline_df['macd_signal'] = macd.macd_signal()
    sma_50 = ta.trend.sma_indicator(baseline_df['close'], window=50)
    sma_200 = ta.trend.sma_indicator(baseline_df['close'], window=200)
    baseline_df['dist_sma_50'] = (baseline_df['close'] - sma_50) / sma_50
    baseline_df['dist_sma_200'] = (baseline_df['close'] - sma_200) / sma_200
    baseline_df.dropna(inplace=True)
    
    macro_input = baseline_df.tail(96).copy()
    macro_input[macro_features] = macro_scaler.transform(macro_input[macro_features])
    x_macro = torch.tensor(macro_input[macro_features].values, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        macro_out = macro_model(x_macro)
        macro_probs = torch.softmax(macro_out, dim=1).squeeze().numpy()
    print(f"Prob: {macro_probs[1]*100:.6f}%")
    
    print("\nPerturbando o ultimo preço em +1.0% (Um movimento forte de 1 min):")
    pert_df = df_1h.copy()
    pert_df.loc[pert_df.index[-1], 'close'] *= 1.01
    
    pert_df['log_return'] = np.log(pert_df['close'] / pert_df['close'].shift(1))
    pert_df['atr_14'] = ta.volatility.average_true_range(pert_df['high'], pert_df['low'], pert_df['close'], window=14)
    pert_df['atr_50'] = ta.volatility.average_true_range(pert_df['high'], pert_df['low'], pert_df['close'], window=50)
    macd = ta.trend.MACD(pert_df['close'])
    pert_df['macd'] = macd.macd()
    pert_df['macd_signal'] = macd.macd_signal()
    sma_50 = ta.trend.sma_indicator(pert_df['close'], window=50)
    sma_200 = ta.trend.sma_indicator(pert_df['close'], window=200)
    pert_df['dist_sma_50'] = (pert_df['close'] - sma_50) / sma_50
    pert_df['dist_sma_200'] = (pert_df['close'] - sma_200) / sma_200
    pert_df.dropna(inplace=True)
    
    macro_input_p = pert_df.tail(96).copy()
    macro_input_p[macro_features] = macro_scaler.transform(macro_input_p[macro_features])
    x_macro_p = torch.tensor(macro_input_p[macro_features].values, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        macro_out_p = macro_model(x_macro_p)
        macro_probs_p = torch.softmax(macro_out_p, dim=1).squeeze().numpy()
    print(f"Prob: {macro_probs_p[1]*100:.6f}%")

if __name__ == '__main__':
    test()
