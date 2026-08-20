import pandas as pd
import numpy as np
import torch
import joblib
import ta
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models.autoformer import Autoformer
from core.risk_manager import RiskManager

def prepare_data(days=60):
    print("Loading data...")
    df_1h = pd.read_parquet('data/macro_1h.parquet')
    df_1m = pd.read_parquet('data/micro_1m.parquet')
    
    max_date = df_1h['timestamp'].max()
    start_date_1h = max_date - pd.Timedelta(days=days+4)
    start_date_1m = max_date - pd.Timedelta(days=days)
    
    df_1h = df_1h[df_1h['timestamp'] >= start_date_1h].copy()
    df_1m = df_1m[df_1m['timestamp'] >= start_date_1m].copy()
    
    # Macro Features
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

def run_simulation():
    df_1h, df_1m = prepare_data(days=60)
    print("Running Macro Inference...")
    macro_preds = run_macro_inference(df_1h)
    
    print("Merging predictions...")
    merged_full = pd.merge_asof(df_1m.sort_values('timestamp'), 
                                macro_preds.sort_values('timestamp'), 
                                on='timestamp', direction='backward')
    merged_full.dropna(subset=['macro_buy', 'macro_sell'], inplace=True)
    
    start_time = merged_full['timestamp'].min()
    end_time = merged_full['timestamp'].max()
    mid_time = start_time + (end_time - start_time) / 2
    
    periods = {
        'Full 60 Days': merged_full,
        'First 30 Days': merged_full[merged_full['timestamp'] <= mid_time].copy(),
        'Last 30 Days': merged_full[merged_full['timestamp'] > mid_time].copy()
    }
    
    scenarios = ['100% USDT', '100% BTC', '50% USDT / 50% BTC']
    
    threshold = 0.60
    max_risk = 0.10
    slippage = 0.0005
    fee_rate = 0.001
    
    results = []
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    axes = axes.flatten()
    
    for ax_idx, (period_name, merged) in enumerate(periods.items()):
        print(f"\nRunning for Period: {period_name}")
        mean_atr = merged['atr_14'].mean()
        std_atr = merged['atr_14'].std()
        
        ax1 = axes[ax_idx]
        ax2 = ax1.twinx()
        
        # Plot BTC Price for reference
        btc_times = merged['timestamp'].iloc[::60]
        btc_prices = merged['close'].iloc[::60]
        ax2.plot(btc_times, btc_prices, color='gray', linestyle='--', alpha=0.5, label='BTC Price (USDT)')
        ax2.set_ylabel("BTC Price (USDT)", color='gray')
        
        colors = {'100% USDT': 'green', '100% BTC': 'orange', '50% USDT / 50% BTC': 'blue'}
        
        for scenario in scenarios:
            print(f"  -> Scenario: {scenario}")
            
            risk_mgr = RiskManager(max_risk_per_trade_pct=max_risk, kelly_fraction=0.5, macro_intensity=3.0)
            risk_mgr.micro_threshold = threshold
            
            start_price = merged.iloc[0].close
            
            if scenario == '100% USDT':
                balance = 1000.0
                btc_held = 0.0
            elif scenario == '100% BTC':
                balance = 0.0
                btc_held = 1000.0 / start_price
            else:
                balance = 500.0
                btc_held = 500.0 / start_price
                
            peak_equity = 1000.0
            max_dd = 0.0
            trades_buy = 0
            trades_sell = 0
            last_trade_time = None
            
            eq_curve = []
            timestamps = []
            
            step_cnt = 0
            for row in merged.itertuples():
                step_cnt += 1
                macro_probs = [row.macro_sell, row.macro_buy]
                micro_probs = [0.49, 0.51]
                
                atr_z = (row.atr_14 - mean_atr) / std_atr if std_atr > 0 else 0
                action, calc_alloc, _ = risk_mgr.evaluate_signal(macro_probs, micro_probs, atr_z)
                
                if action == "Buy" and macro_probs[1] < threshold:
                    action = "Hold"
                    calc_alloc = 0.0
                elif action == "Sell" and macro_probs[0] < threshold:
                    action = "Hold"
                    calc_alloc = 0.0
                
                if action != "Hold" and last_trade_time is not None:
                    if (row.timestamp - last_trade_time).total_seconds() < 60:
                        action = "Hold"
                        calc_alloc = 0.0
                        
                if action == "Buy" and balance > 1.0:
                    fiat_spend = balance * calc_alloc
                    if fiat_spend >= 5.0:
                        limit_price = row.close * (1 + slippage)
                        btc_bought = (fiat_spend / limit_price) * (1 - fee_rate)
                        balance -= fiat_spend
                        btc_held += btc_bought
                        trades_buy += 1
                        last_trade_time = row.timestamp
                        
                elif action == "Sell" and (btc_held * row.close) > 1.0:
                    btc_to_sell = btc_held * calc_alloc
                    fiat_value = btc_to_sell * row.close
                    
                    if fiat_value >= 5.0:
                        limit_price = row.close * (1 - slippage)
                        usdt_gained = (btc_to_sell * limit_price) * (1 - fee_rate)
                        btc_held -= btc_to_sell
                        balance += usdt_gained
                        trades_sell += 1
                        last_trade_time = row.timestamp
                        
                equity = balance + (btc_held * row.close)
                if equity > peak_equity:
                    peak_equity = equity
                dd = (peak_equity - equity) / peak_equity
                if dd > max_dd:
                    max_dd = dd
                    
                if step_cnt % 60 == 0:
                    eq_curve.append(equity)
                    timestamps.append(row.timestamp)
            
            final_equity = balance + (btc_held * merged.iloc[-1].close)
            roi = ((final_equity / 1000.0) - 1.0) * 100
            
            bnh_roi = ((merged.iloc[-1].close / start_price) - 1.0) * 100
            
            ax1.plot(timestamps, eq_curve, color=colors[scenario], label=f"{scenario} (ROI: {roi:.2f}%)")
            
            results.append({
                'Period': period_name,
                'Start_Allocation': scenario,
                'Final_Equity': final_equity,
                'ROI_Pct': roi,
                'BnH_ROI_Pct': bnh_roi,
                'Max_Drawdown_Pct': max_dd * 100,
                'Trades_Buy': trades_buy,
                'Trades_Sell': trades_sell,
                'Total_Trades': trades_buy + trades_sell
            })
            
        ax1.set_title(f"Teste de Robustez - {period_name}")
        ax1.set_ylabel("Patrimônio Total ($)")
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("robustness_curves.png")
    plt.close()
    
    res_df = pd.DataFrame(results)
    res_df.to_csv("robustness_results.csv", index=False)
    print("Results saved to robustness_results.csv and robustness_curves.png")

if __name__ == "__main__":
    run_simulation()
