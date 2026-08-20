import ccxt
import pandas as pd
import numpy as np
import torch
import joblib
import time
from datetime import datetime, timedelta
import os
import sys

from model import TimeSeriesTransformer
from data_collection import feature_engineering

def run_backtest():
    symbol = 'BTC/USDT'
    timeframe = '15m'
    
    exchange = ccxt.bybit({'enableRateLimit': True})
    
    print("Fetching data for the last 30 days...")
    # Fetch 30 days of data + buffer for indicators
    # 30 days * 96 candles/day = 2880. + 300 buffer = 3180 candles
    total_candles = 3200
    all_ohlcv = []
    since = exchange.milliseconds() - total_candles * 15 * 60 * 1000
    
    while len(all_ohlcv) < total_candles:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if len(ohlcv) == 0:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
        time.sleep(0.1)
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    # Restrict strictly to last 30 days of features, but we need raw data earlier for SMAs
    thirty_days_ago = df.index[-1] - timedelta(days=30)
    
    print("Applying feature engineering...")
    df_features = feature_engineering(df.copy())
    df_features.dropna(inplace=True)
    
    print("Loading model and scaler...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TimeSeriesTransformer(num_features=12).to(device)
    model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
    model.eval()
    
    scaler = joblib.load('models/scaler.pkl')
    
    features_list = [
        'log_return', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
        'bb_width', 'bb_percent', 'atr', 'volume_ratio'
    ]
    
    df_scaled = df_features.copy()
    df_scaled[features_list] = scaler.transform(df_features[features_list])
    
    seq_len = 60
    
    # Backtest simulation vars
    initial_balance = 1000.0
    balance_usdt = initial_balance
    balance_btc = 0.0
    trade_amount = 100.0 # Fixed $100 per trade
    fee_rate = 0.001 # 0.1%
    
    buy_count = 0
    sell_count = 0
    hold_count = 0
    
    trade_log = []
    
    # Start iterating from the point where we have a full sequence and it's within the last 30 days
    start_idx = seq_len
    
    for i in range(start_idx, len(df_scaled)):
        current_time = df_scaled.index[i]
        
        # Only start trading if within the last 30 days
        if current_time < thirty_days_ago:
            continue
            
        # The sequence uses data from i-seq_len to i-1
        seq_data = df_scaled[features_list].values[i-seq_len:i]
        x_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze()
            
            # probs[0] = Probability of Sell, probs[1] = Probability of Buy
            prob_sell = probs[0].item()
            prob_buy = probs[1].item()
            
            confidence_threshold = 0.60
            
            if prob_buy >= confidence_threshold:
                pred = 1 # Buy
            elif prob_sell >= confidence_threshold:
                pred = 0 # Sell
            else:
                pred = -1 # Hold (forced)
            
        current_price = df_features['close'].iloc[i]
        decision = "Hold"
        
        if pred == 1 and balance_usdt > 0:
            decision = "Buy"
            amount = min(trade_amount, balance_usdt)
            btc_bought = (amount * (1 - fee_rate)) / current_price
            balance_usdt -= amount
            balance_btc += btc_bought
            buy_count += 1
            
            trade_log.append({
                'Time': current_time,
                'Action': 'BUY',
                'Price': current_price,
                'Value_USDT': amount,
                'BTC_Amount': btc_bought
            })
            
        elif pred == 0 and balance_btc > 0:
            decision = "Sell"
            btc_val = balance_btc * current_price
            amount = min(trade_amount, btc_val)
            btc_to_sell = amount / current_price
            usdt_received = amount * (1 - fee_rate)
            balance_btc -= btc_to_sell
            balance_usdt += usdt_received
            sell_count += 1
            
            trade_log.append({
                'Time': current_time,
                'Action': 'SELL',
                'Price': current_price,
                'Value_USDT': usdt_received,
                'BTC_Amount': btc_to_sell
            })
            
        else:
            hold_count += 1
            
    final_portfolio_value = balance_usdt + (balance_btc * df_features['close'].iloc[-1])
    profit = final_portfolio_value - initial_balance
    roi = (profit / initial_balance) * 100
    
    # Output Results
    print("="*40)
    print("BACKTEST RESULTS (LAST 30 DAYS)")
    print("="*40)
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${final_portfolio_value:.2f}")
    print(f"Net Profit:      ${profit:.2f}")
    print(f"ROI:             {roi:.2f}%")
    print(f"Total Buys:      {buy_count}")
    print(f"Total Sells:     {sell_count}")
    print(f"Total Holds:     {hold_count}")
    
    print("\nSAVING TRADES LOG TO CSV...")
    if trade_log:
        log_df = pd.DataFrame(trade_log)
        log_df.to_csv("last_month_trades.csv", index=False)
        print("Saved to last_month_trades.csv")
    else:
        print("No trades executed.")

if __name__ == "__main__":
    run_backtest()
