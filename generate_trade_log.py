import pandas as pd
import numpy as np
import torch
import joblib
import ta
import os
from models.autoformer import Autoformer
from core.risk_manager import RiskManager

def prepare_data(days=60):
    df_1h = pd.read_parquet('data/macro_1h.parquet')
    df_1m = pd.read_parquet('data/micro_1m.parquet')
    max_date = df_1h['timestamp'].max()
    start_date_1h = max_date - pd.Timedelta(days=days+4)
    start_date_1m = max_date - pd.Timedelta(days=days)
    df_1h = df_1h[df_1h['timestamp'] >= start_date_1h].copy()
    df_1m = df_1m[df_1m['timestamp'] >= start_date_1m].copy()
    
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
    return df_1h, df_1m

def run_macro_inference(df_1h):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    macro_scaler = joblib.load('models/macro_scaler.pkl')
    features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
    model = Autoformer(num_features=len(features), num_classes=2).to(device)
    model.load_state_dict(torch.load('models/macro_autoformer.pt', map_location=device))
    model.eval()
    
    seq_len = 96
    inputs = macro_scaler.transform(df_1h[features])
    X = []
    for i in range(len(inputs) - seq_len + 1):
        X.append(inputs[i:i+seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    valid_timestamps = df_1h['timestamp'].iloc[seq_len-1:].values
    valid_atr = df_1h['atr_14'].iloc[seq_len-1:].values
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch[0].to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.extend(probs)
    all_probs = np.array(all_probs)
    res_df = pd.DataFrame({
        'timestamp': valid_timestamps,
        'macro_sell': all_probs[:, 0],
        'macro_buy': all_probs[:, 1],
        'atr_14': valid_atr
    })
    return res_df

def run_log_simulation():
    df_1h, df_1m = prepare_data(days=60)
    macro_preds = run_macro_inference(df_1h)
    
    merged = pd.merge_asof(df_1m.sort_values('timestamp'), 
                           macro_preds.sort_values('timestamp'), 
                           on='timestamp', direction='backward')
    merged.dropna(subset=['macro_buy', 'macro_sell'], inplace=True)
    
    threshold = 0.60
    max_risk = 0.10
    slippage = 0.0005
    fee_rate = 0.001
    
    mean_atr = merged['atr_14'].mean()
    std_atr = merged['atr_14'].std()
    
    risk_mgr = RiskManager(max_risk_per_trade_pct=max_risk, kelly_fraction=0.5, macro_intensity=3.0)
    risk_mgr.micro_threshold = threshold
    
    balance = 1000.0
    btc_held = 0.0
    last_trade_time = None
    
    trade_log = []
    
    for row in merged.itertuples():
        macro_probs = [row.macro_sell, row.macro_buy]
        micro_probs = [0.49, 0.51]
        
        atr_z = (row.atr_14 - mean_atr) / std_atr if std_atr > 0 else 0
        action, calc_alloc, _ = risk_mgr.evaluate_signal(macro_probs, micro_probs, atr_z)
        
        if action == "Buy" and macro_probs[1] < threshold:
            action = "Hold"
        elif action == "Sell" and macro_probs[0] < threshold:
            action = "Hold"
            
        if action != "Hold" and last_trade_time is not None:
            if (row.timestamp - last_trade_time).total_seconds() < 60:
                action = "Hold"
                
        if action == "Buy" and balance > 1.0:
            fiat_spend = balance * calc_alloc
            if fiat_spend >= 5.0:
                limit_price = row.close * (1 + slippage)
                btc_bought = (fiat_spend / limit_price) * (1 - fee_rate)
                balance -= fiat_spend
                btc_held += btc_bought
                last_trade_time = row.timestamp
                
                trade_log.append({
                    'Timestamp': row.timestamp,
                    'Action': 'BUY',
                    'Price_USDT': limit_price,
                    'Amount_BTC': btc_bought,
                    'Value_USDT': fiat_spend,
                    'Balance_USDT': balance,
                    'Balance_BTC': btc_held
                })
                
        elif action == "Sell" and (btc_held * row.close) > 1.0:
            btc_to_sell = btc_held * calc_alloc
            fiat_value = btc_to_sell * row.close
            
            if fiat_value >= 5.0:
                limit_price = row.close * (1 - slippage)
                usdt_gained = (btc_to_sell * limit_price) * (1 - fee_rate)
                btc_held -= btc_to_sell
                balance += usdt_gained
                last_trade_time = row.timestamp
                
                trade_log.append({
                    'Timestamp': row.timestamp,
                    'Action': 'SELL',
                    'Price_USDT': limit_price,
                    'Amount_BTC': btc_to_sell,
                    'Value_USDT': usdt_gained,
                    'Balance_USDT': balance,
                    'Balance_BTC': btc_held
                })
                
    res_df = pd.DataFrame(trade_log)
    res_df.to_csv("transaction_log.csv", index=False)
    
if __name__ == "__main__":
    run_log_simulation()
