import os
import sys
import time
import pandas as pd
import numpy as np
import ta
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt

from models.autoformer import Autoformer
from models.cnn_transformer import CNNTransformer
from core.risk_manager import RiskManager

def build_volume_bars(df_1m, target_volume=5.0):
    vol_bars = []
    current_vol = 0
    if len(df_1m) == 0: return pd.DataFrame()
        
    bar_open = df_1m['open'].iloc[0]
    bar_high = df_1m['high'].iloc[0]
    bar_low = df_1m['low'].iloc[0]
    timestamp = df_1m.index[0]
    
    for idx, row in df_1m.iterrows():
        current_vol += row['volume']
        bar_high = max(bar_high, row['high'])
        bar_low = min(bar_low, row['low'])
        
        if current_vol >= target_volume:
            vol_bars.append({
                'timestamp': timestamp,
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
            timestamp = idx
            
    df_vol = pd.DataFrame(vol_bars)
    if len(df_vol) > 0:
        df_vol.set_index('timestamp', inplace=True)
    return df_vol

def calc_max_drawdown(equity_series):
    roll_max = equity_series.cummax()
    drawdown = equity_series / roll_max - 1.0
    return drawdown.min()

def run_backtest(symbol='BTCUSDT'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"1. Carregando dados de 1 hora do disco (OOS 60 dias) para {symbol}...")
    df_1h = pd.read_parquet(f'data/macro_1h_{symbol}.parquet')
    df_1h.set_index('timestamp', inplace=True)
    df_1h = df_1h.tail(24 * 65) 
    
    print(f"2. Carregando dados de 1 minuto do disco (OOS 60 dias) para {symbol}...")
    df_1m = pd.read_parquet(f'data/micro_1m_{symbol}.parquet')
    df_1m.set_index('timestamp', inplace=True)
    df_1m = df_1m.tail(60 * 24 * 60) 
    
    print("3. Construindo Volume Bars...")
    df_vol = build_volume_bars(df_1m, target_volume=5.0)
    
    print("4. Feature Engineering Macro...")
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
    
    print("5. Feature Engineering Micro...")
    if len(df_vol) > 0:
        df_vol['log_return'] = np.log(df_vol['close'] / df_vol['close'].shift(1))
        df_vol['momentum'] = df_vol['close'] - df_vol['close'].shift(1)
        df_vol.dropna(inplace=True)
    else:
        print("Erro: df_vol vazio.")
        return
    
    macro_scaler = joblib.load(f'models/macro_scaler_{symbol}.pkl')
    micro_scaler = joblib.load('models/micro_scaler.pkl')
    
    macro_features = ['log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 'dist_sma_50', 'dist_sma_200']
    micro_features = ['log_return', 'momentum']
    
    print("6. Carregando Modelos...")
    macro_model = Autoformer(num_features=len(macro_features), num_classes=2).to(device)
    macro_model.load_state_dict(torch.load(f'models/macro_autoformer_{symbol}.pt', map_location=device))
    macro_model.eval()
    
    micro_model = CNNTransformer(num_features=len(micro_features), num_classes=2).to(device)
    micro_model.load_state_dict(torch.load('models/micro_cnn.pt', map_location=device))
    micro_model.eval()
    
    allocations = [0.01, 0.03, 0.05, 0.10]
    
    macro_scaled = df_1h.copy()
    macro_scaled[macro_features] = macro_scaler.transform(df_1h[macro_features])
    
    micro_scaled = df_vol.copy()
    micro_scaled[micro_features] = micro_scaler.transform(df_vol[micro_features])
    
    macro_seq_len = 96
    micro_seq_len = 10
    
    micro_idx = micro_scaled.index
    macro_idx = macro_scaled.index
    
    results = {}
    
    print(f"\n7. Rodando Backtest ao Longo do Tempo para {symbol}...")
    
    for alloc_pct in allocations:
        print(f"--- TESTANDO ALOCACAO: {alloc_pct*100:.0f}% ---")
        risk_manager = RiskManager(max_risk_per_trade_pct=alloc_pct, kelly_fraction=0.5)
        risk_manager.micro_threshold = 0.60
        
        balance_usdt = 1000.0
        balance_btc = 0.0
        
        fee_rate = 0.001
        slippage = 0.0005
        
        equity_curve = []
        trades_buy = 0
        trades_sell = 0
        
        # Win rate tracking (simplificado: track average buy price, compare at sell)
        win_trades = 0
        loss_trades = 0
        avg_buy_price = 0.0
        total_btc_bought_for_calc = 0.0
        
        for i in range(micro_seq_len, len(micro_idx)):
            current_time = micro_idx[i]
            current_price = df_vol['close'].iloc[i]
            
            # Record equity every step
            current_equity = balance_usdt + (balance_btc * current_price)
            equity_curve.append({'time': current_time, 'equity': current_equity, 'btc_price': current_price})
            
            available_macro = macro_idx[macro_idx <= current_time]
            if len(available_macro) < macro_seq_len:
                continue
                
            latest_macro_time = available_macro[-1]
            macro_idx_pos = macro_scaled.index.get_loc(latest_macro_time)
            
            macro_seq = macro_scaled[macro_features].iloc[macro_idx_pos - macro_seq_len + 1 : macro_idx_pos + 1].values
            micro_seq = micro_scaled[micro_features].iloc[i - micro_seq_len + 1 : i + 1].values
            
            if len(macro_seq) != macro_seq_len or len(micro_seq) != micro_seq_len:
                continue
                
            x_macro = torch.tensor(macro_seq, dtype=torch.float32).unsqueeze(0).to(device)
            x_micro = torch.tensor(micro_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                macro_out = macro_model(x_macro)
                macro_probs = torch.softmax(macro_out, dim=1).squeeze().cpu().numpy().tolist()
                micro_out = micro_model(x_micro)
                micro_probs = torch.softmax(micro_out, dim=1).squeeze().cpu().numpy().tolist()
                
            atr_volatility = 1.0 
            if 'atr_14' in df_1h.columns:
                recent_atrs = df_1h['atr_14'].iloc[macro_idx_pos-14 : macro_idx_pos]
                if len(recent_atrs) > 2:
                    mean_atr = recent_atrs.mean()
                    std_atr = recent_atrs.std()
                    if std_atr > 0:
                        atr_volatility = (df_1h['atr_14'].iloc[macro_idx_pos] - mean_atr) / std_atr
                        
            action, alloc, reason = risk_manager.evaluate_signal(macro_probs, micro_probs, atr_volatility)
            
            if action == "Buy" and balance_usdt > 10:
                usdt_spend = balance_usdt * alloc
                limit_price = current_price * (1 + slippage)
                btc_bought = (usdt_spend * (1 - fee_rate)) / limit_price
                balance_usdt -= usdt_spend
                balance_btc += btc_bought
                trades_buy += 1
                
                # Tracking para win rate
                total_cost = usdt_spend
                avg_buy_price = ((avg_buy_price * total_btc_bought_for_calc) + total_cost) / (total_btc_bought_for_calc + btc_bought)
                total_btc_bought_for_calc += btc_bought
                
            elif action == "Sell" and balance_btc > 0.0001:
                btc_val = balance_btc * current_price
                usdt_spend = btc_val * alloc
                btc_to_sell = usdt_spend / current_price
                limit_price = current_price * (1 - slippage)
                usdt_gained = (btc_to_sell * limit_price) * (1 - fee_rate)
                balance_btc -= btc_to_sell
                balance_usdt += usdt_gained
                trades_sell += 1
                
                # Win rate math
                if avg_buy_price > 0:
                    realized_pnl = usdt_gained - (btc_to_sell * avg_buy_price)
                    if realized_pnl > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    total_btc_bought_for_calc -= btc_to_sell
                    if total_btc_bought_for_calc <= 0.0001:
                        avg_buy_price = 0.0
                        total_btc_bought_for_calc = 0.0
                
        df_eq = pd.DataFrame(equity_curve)
        if len(df_eq) > 0:
            df_eq.set_index('time', inplace=True)
            
            final_equity = df_eq['equity'].iloc[-1]
            roi = ((final_equity / 1000.0) - 1.0) * 100
            mdd = calc_max_drawdown(df_eq['equity']) * 100
            
            total_trades = trades_buy + trades_sell
            win_rate = (win_trades / (win_trades + loss_trades)) * 100 if (win_trades + loss_trades) > 0 else 0.0
            
            results[alloc_pct] = {
                'df': df_eq,
                'roi': roi,
                'mdd': mdd,
                'trades_buy': trades_buy,
                'trades_sell': trades_sell,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'final_equity': final_equity
            }
        
    print("\n8. Gerando Relatório e Gráficos...")
    
    if len(results) == 0:
        print("Sem trades ou dados suficientes para gerar relatório.")
        return

    plt.figure(figsize=(14, 8))
    
    # Plot baseline (normalized to 1000)
    baseline_df = results[allocations[0]]['df']
    btc_initial = baseline_df['btc_price'].iloc[0]
    baseline_df['btc_norm'] = (baseline_df['btc_price'] / btc_initial) * 1000.0
    
    plt.plot(baseline_df.index, baseline_df['btc_norm'], label=f'{symbol} Buy & Hold', color='black', linewidth=2, linestyle='--')
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    report_text = f"# 📊 Relatório Quantitativo OOS (60 Dias) - {symbol}\n\n"
    
    for i, alloc_pct in enumerate(allocations):
        if alloc_pct in results:
            r = results[alloc_pct]
            plt.plot(r['df'].index, r['df']['equity'], label=f'Alocação {alloc_pct*100:.0f}% (ROI: {r["roi"]:.2f}%)', color=colors[i], linewidth=2)
            
            report_text += f"### Alocação de Risco: {alloc_pct*100:.0f}%\n"
            report_text += f"- **Saldo Final**: ${r['final_equity']:.2f}\n"
            report_text += f"- **Lucro Líquido (ROI)**: {r['roi']:.2f}%\n"
            report_text += f"- **Max Drawdown**: {r['mdd']:.2f}%\n"
            report_text += f"- **Total de Ordens**: {r['total_trades']} ({r['trades_buy']} Compras / {r['trades_sell']} Vendas)\n"
            report_text += f"- **Win Rate (Ciclos de Lucro)**: {r['win_rate']:.2f}%\n\n"
        
    plt.title(f'Curva de Capital Híbrida V2.3 (Out-Of-Sample) vs {symbol}', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Saldo ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'backtest_chart_{symbol}.png')
    
    with open(f'v2_backtest_report_{symbol}.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"Gráfico salvo em 'backtest_chart_{symbol}.png'!")
    print(f"Relatório salvo em 'v2_backtest_report_{symbol}.md'!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_backtest(sys.argv[1])
    else:
        run_backtest('BTCUSDT')
