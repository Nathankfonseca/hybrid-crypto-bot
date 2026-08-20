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
    # Need 4 extra days for the 96h sequence (Autoformer context)
    start_date_1h = max_date - pd.Timedelta(days=days+4)
    start_date_1m = max_date - pd.Timedelta(days=days)
    
    df_1h = df_1h[df_1h['timestamp'] >= start_date_1h].copy()
    df_1m = df_1m[df_1m['timestamp'] >= start_date_1m].copy()
    
    print(f"Data filtered. 1H: {len(df_1h)} rows, 1M: {len(df_1m)} rows.")
    
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
    print("Running Macro Inference...")
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

def run_grid_search():
    df_1h, df_1m = prepare_data(days=60)
    macro_preds = run_macro_inference(df_1h)
    
    print("Merging predictions with 1M execution data...")
    merged = pd.merge_asof(df_1m.sort_values('timestamp'), 
                           macro_preds.sort_values('timestamp'), 
                           on='timestamp', direction='backward')
    merged.dropna(subset=['macro_buy', 'macro_sell'], inplace=True)
    
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
    allocations = [0.01, 0.03, 0.05, 0.10, 0.20, 0.50, 1.00]
    
    mean_atr = merged['atr_14'].mean()
    std_atr = merged['atr_14'].std()
    
    slippage = 0.0005
    fee_rate = 0.001
    
    results = []
    equity_curves = {}
    
    total_sims = len(thresholds) * len(allocations)
    sim_idx = 0
    
    for thresh in thresholds:
        for alloc in allocations:
            sim_idx += 1
            print(f"[{sim_idx}/{total_sims}] Simulating Threshold: {thresh} | Max Risk: {alloc*100}%")
            
            risk_mgr = RiskManager(max_risk_per_trade_pct=alloc, kelly_fraction=0.5, macro_intensity=3.0)
            risk_mgr.micro_threshold = thresh
            
            balance = 1000.0 # Paper money simulated in app
            btc_held = 0.0
            peak_equity = 1000.0
            max_dd = 0.0
            trades_buy = 0
            trades_sell = 0
            
            # For 1M cooldown
            last_trade_time = None
            
            eq_curve = []
            timestamps = []
            
            step_cnt = 0
            for row in merged.itertuples():
                step_cnt += 1
                macro_probs = [row.macro_sell, row.macro_buy]
                micro_probs = [0.49, 0.51] # Hardcoded in hybrid app
                
                atr_z = (row.atr_14 - mean_atr) / std_atr if std_atr > 0 else 0
                action, calc_alloc, _ = risk_mgr.evaluate_signal(macro_probs, micro_probs, atr_z)
                
                # Strict certainty rule (as implemented in hybrid_desktop_app)
                if action == "Buy" and macro_probs[1] < thresh:
                    action = "Hold"
                    calc_alloc = 0.0
                elif action == "Sell" and macro_probs[0] < thresh:
                    action = "Hold"
                    calc_alloc = 0.0
                
                # Cooldown rule (Uma ordem por minuto)
                if action != "Hold" and last_trade_time is not None:
                    # diff in seconds
                    diff = (row.timestamp - last_trade_time).total_seconds()
                    if diff < 60:
                        action = "Hold"
                        calc_alloc = 0.0
                        
                # Execution
                if action == "Buy" and balance > 1.0:
                    fiat_spend = balance * calc_alloc
                    if fiat_spend < 5.0:
                        fiat_spend = 0.0 # Ignore logic (removed minimum order bypass)
                    
                    if fiat_spend > 0:
                        limit_price = row.close * (1 + slippage)
                        btc_bought = (fiat_spend / limit_price) * (1 - fee_rate)
                        balance -= fiat_spend
                        btc_held += btc_bought
                        trades_buy += 1
                        last_trade_time = row.timestamp
                        
                elif action == "Sell" and (btc_held * row.close) > 1.0:
                    btc_to_sell = btc_held * calc_alloc
                    fiat_value = btc_to_sell * row.close
                    
                    if fiat_value < 5.0:
                        btc_to_sell = 0.0 # Ignore logic
                        
                    if btc_to_sell > 0:
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
                    
                # Downsample equity curve saving for memory performance (save every 1 hour ~ 60 mins)
                if step_cnt % 60 == 0:
                    eq_curve.append(equity)
                    timestamps.append(row.timestamp)
            
            final_equity = balance + (btc_held * merged.iloc[-1].close)
            roi = ((final_equity / 1000.0) - 1.0) * 100
            
            label = f"T:{thresh}-R:{alloc*100}%"
            equity_curves[label] = {'times': timestamps, 'equity': eq_curve, 'roi': roi}
            
            results.append({
                'Threshold': thresh,
                'Max_Risk_Pct': alloc * 100,
                'Final_Equity': final_equity,
                'ROI_Pct': roi,
                'Max_Drawdown_Pct': max_dd * 100,
                'Trades_Buy': trades_buy,
                'Trades_Sell': trades_sell,
                'Total_Trades': trades_buy + trades_sell
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv("grid_search_results.csv", index=False)
    print("Results saved to grid_search_results.csv")
    
    # 1. Plot Heatmap
    plt.figure(figsize=(10, 8))
    pivot = res_df.pivot(index='Threshold', columns='Max_Risk_Pct', values='ROI_Pct')
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title("ROI (%) por Limiar e Risco Máximo (60 Dias)")
    plt.savefig("heatmap_roi.png")
    plt.close()
    
    # 2. Plot Top 5 Equity Curves
    fig, ax1 = plt.subplots(figsize=(12, 6))
    top_5 = res_df.sort_values('ROI_Pct', ascending=False).head(5)
    
    for _, row in top_5.iterrows():
        label = f"T:{row['Threshold']}-R:{row['Max_Risk_Pct']}%"
        data = equity_curves[label]
        ax1.plot(data['times'], data['equity'], label=f"{label} (ROI: {data['roi']:.2f}%)")
        
    ax1.set_title("Top 5 Configurações - Curva de Patrimônio (60 Dias)")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Patrimônio ($)")
    
    ax2 = ax1.twinx()
    btc_times = merged['timestamp'].iloc[::60]
    btc_prices = merged['close'].iloc[::60]
    ax2.plot(btc_times, btc_prices, color='gray', linestyle='--', alpha=0.5, label='Bitcoin Price (USDT)')
    ax2.set_ylabel("BTC Price (USDT)", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    ax1.grid(True, alpha=0.3)
    plt.savefig("top5_equity_curve.png")
    plt.close()
    
    print("Plots saved: heatmap_roi.png and top5_equity_curve.png")

if __name__ == "__main__":
    run_grid_search()
