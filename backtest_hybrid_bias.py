import pandas as pd
import numpy as np
import torch
import joblib
import ta
from models.autoformer import Autoformer
from models.micro_mlp import MicroMLP
from core.risk_manager import RiskManager
import os
import json

def fetch_synthetic_volume_bars(df_1m, target_volume=50.0):
    vol_bars = []
    current_vol = 0
    if len(df_1m) == 0: return pd.DataFrame()
    bar_open = df_1m['open'].iloc[0]
    bar_high = df_1m['high'].iloc[0]
    bar_low = df_1m['low'].iloc[0]
    
    for row in df_1m.itertuples():
        current_vol += row.volume
        bar_high = max(bar_high, row.high)
        bar_low = min(bar_low, row.low)
        if current_vol >= target_volume:
            vol_bars.append({
                'timestamp': row.timestamp,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': row.close,
                'volume': current_vol
            })
            current_vol = 0
            bar_open = row.close
            bar_high = row.close
            bar_low = row.close
    return pd.DataFrame(vol_bars)

def prepare_data():
    print("Preparing Micro Data...")
    df_1m = pd.read_parquet('data/micro_1m.parquet')
    df_vol = fetch_synthetic_volume_bars(df_1m)
    df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
    df_vol['momentum'] = df_vol['close'] - df_vol['close'].shift(1)
    df_vol.dropna(inplace=True)
    
    micro_seq_len = 10
    micro_features = []
    for i in range(micro_seq_len):
        df_vol[f'log_return_lag_{i}'] = df_vol['log_return'].shift(i)
        df_vol[f'momentum_lag_{i}'] = df_vol['momentum'].shift(i)
        micro_features.append(f'log_return_lag_{i}')
        micro_features.append(f'momentum_lag_{i}')
    df_vol.dropna(inplace=True)
    
    print("Preparing Macro Data...")
    df_1h = pd.read_parquet('data/macro_1h.parquet')
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
    
    return df_vol, df_1h, micro_features

def batched_inference(df, model, scaler, features, seq_len, is_macro, device):
    inputs = scaler.transform(df[features])
    if is_macro:
        # Create rolling sequences of length seq_len
        X = []
        for i in range(len(inputs) - seq_len + 1):
            X.append(inputs[i:i+seq_len])
        X = torch.tensor(np.array(X), dtype=torch.float32)
        valid_timestamps = df['timestamp'].iloc[seq_len-1:].values
        valid_atr = df['atr_14'].iloc[seq_len-1:].values
    else:
        # Micro MLP takes 1 row directly (20 features)
        X = torch.tensor(inputs, dtype=torch.float32)
        valid_timestamps = df['timestamp'].values
        valid_atr = None
        
    dataset = torch.utils.data.TensorDataset(X)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=False)
    
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch[0].to(device))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.extend(probs)
            
    all_probs = np.array(all_probs)
    
    res_df = pd.DataFrame({
        'timestamp': valid_timestamps,
        'prob_sell': all_probs[:, 0],
        'prob_buy': all_probs[:, 1]
    })
    
    if is_macro:
        res_df['atr_14'] = valid_atr
        
    return res_df

def run_backtest():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_vol, df_1h, micro_features = prepare_data()
    
    macro_scaler = joblib.load('models/macro_scaler.pkl')
    micro_scaler = joblib.load('models/micro_mlp_scaler.pkl')
    macro_features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
    
    macro_model = Autoformer(num_features=len(macro_features), num_classes=2).to(device)
    macro_model.load_state_dict(torch.load('models/macro_autoformer.pt', map_location=device))
    macro_model.eval()
    
    micro_model = MicroMLP(num_features=len(micro_features)).to(device)
    micro_model.load_state_dict(torch.load('models/micro_mlp.pt', map_location=device))
    micro_model.eval()
    
    # Take last 60 days of volume bars (~7200 bars at 50 BTC per bar)
    df_vol_oos = df_vol.tail(7200).copy()
    
    print("Running Macro Inference...")
    macro_preds = batched_inference(df_1h, macro_model, macro_scaler, macro_features, 96, True, device)
    
    print("Running Micro Inference...")
    micro_preds = batched_inference(df_vol_oos, micro_model, micro_scaler, micro_features, 10, False, device)
    
    print("Merging predictions...")
    macro_preds.rename(columns={'prob_sell': 'macro_sell', 'prob_buy': 'macro_buy'}, inplace=True)
    micro_preds.rename(columns={'prob_sell': 'micro_sell', 'prob_buy': 'micro_buy'}, inplace=True)
    
    # As of merge: match each micro timestamp with the most recent macro timestamp
    merged = pd.merge_asof(micro_preds.sort_values('timestamp'), 
                           macro_preds.sort_values('timestamp'), 
                           on='timestamp', direction='backward')
    
    # Add actual prices for simulation
    merged = pd.merge(merged, df_vol_oos[['timestamp', 'close']], on='timestamp', how='left')
    merged.dropna(inplace=True)
    
    intensities = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    results = []
    
    slippage = 0.0005
    fee_rate = 0.001
    
    mean_atr = merged['atr_14'].mean()
    std_atr = merged['atr_14'].std()
    
    print("\nStarting Simulation over Intensities...")
    
    for intensity in intensities:
        risk_mgr = RiskManager(macro_intensity=intensity)
        
        balance = 10000.0
        btc_held = 0.0
        peak_equity = 10000.0
        max_dd = 0.0
        trades = 0
        last_trade_time = None
        
        for row in merged.itertuples():
            macro_probs = [row.macro_sell, row.macro_buy]
            micro_probs = [row.micro_sell, row.micro_buy]
            
            atr_z = (row.atr_14 - mean_atr) / std_atr if std_atr > 0 else 0
            action, alloc, _ = risk_mgr.evaluate_signal(macro_probs, micro_probs, atr_z)
            
            # Cooldown logic
            if action != "Hold" and last_trade_time == row.timestamp:
                action = "Hold"
                
            if action == "Buy" and balance > 10:
                usdt_spend = balance * alloc
                if usdt_spend > 10:
                    limit_price = row.close * (1 + slippage)
                    btc_bought = (usdt_spend / limit_price) * (1 - fee_rate)
                    balance -= usdt_spend
                    btc_held += btc_bought
                    trades += 1
                    last_trade_time = row.timestamp
            elif action == "Sell" and (btc_held * row.close) > 10:
                btc_to_sell = btc_held * alloc
                if (btc_to_sell * row.close) > 10:
                    limit_price = row.close * (1 - slippage)
                    usdt_gained = (btc_to_sell * limit_price) * (1 - fee_rate)
                    btc_held -= btc_to_sell
                    balance += usdt_gained
                    trades += 1
                    last_trade_time = row.timestamp
                    
            equity = balance + (btc_held * row.close)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity
            if dd > max_dd:
                max_dd = dd
                
        final_equity = balance + (btc_held * merged.iloc[-1]['close'])
        roi = ((final_equity - 10000) / 10000) * 100
        
        print(f"Intensity {intensity:.1f} -> ROI: {roi:.2f}% | MaxDD: {max_dd*100:.2f}% | Trades: {trades}")
        results.append({
            "Intensity": intensity,
            "ROI_pct": roi,
            "MaxDD_pct": max_dd * 100,
            "Trades": trades
        })
        
    with open("hybrid_bias_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == '__main__':
    run_backtest()
