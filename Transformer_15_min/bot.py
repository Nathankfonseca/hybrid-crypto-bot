import ccxt
import pandas as pd
import numpy as np
import torch
import time
import joblib
import os
from datetime import datetime
from dotenv import load_dotenv

from model import TimeSeriesTransformer
from data_collection import feature_engineering # Reuse the function

# Load environment variables
load_dotenv()
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_SECRET')

# Configuration
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
SEQUENCE_LENGTH = 60
NUM_FEATURES = 12

def init_exchange():
    exchange = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        # 'options': {'defaultType': 'future'} # Uncomment if trading futures
    })
    return exchange

def load_model_and_scaler(device):
    print("Loading scaler and model...")
    scaler = joblib.load('models/scaler.pkl')
    
    model = TimeSeriesTransformer(num_features=NUM_FEATURES).to(device)
    model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
    model.eval()
    
    return model, scaler

def get_latest_data(exchange):
    # Fetch enough candles to compute indicators + sequence length
    # RSI 14, SMA 200, MACD etc. require at least 200 candles before they are valid.
    # We need 200 + 60 = 260. Let's fetch 300 to be safe.
    limit = 300
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def run_bot():
    if not API_KEY or not API_SECRET:
        print("Warning: API Keys not found in .env. The bot will run in 'Dry-Run' mode (no orders will be sent).")
        dry_run = True
    else:
        dry_run = False
        print("API Keys found. LIVE TRADING MODE.")

    exchange = init_exchange()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scaler = load_model_and_scaler(device)
    
    features_list = [
        'log_return', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
        'bb_width', 'bb_percent', 'atr', 'volume_ratio'
    ]
    
    last_processed_candle = None
    
    print(f"[{datetime.now()}] Bot started. Waiting for new {TIMEFRAME} candles on {SYMBOL}...")
    
    while True:
        try:
            # Fetch latest data
            df = get_latest_data(exchange)
            
            # The last candle in ccxt is usually the current *incomplete* candle.
            # We want to make a decision based on the last *closed* candle.
            # Let's drop the last row.
            df = df.iloc[:-1]
            
            latest_candle_time = df.index[-1]
            
            if last_processed_candle == latest_candle_time:
                # We already processed this candle, wait a bit and check again
                time.sleep(30)
                continue
                
            print(f"[{datetime.now()}] New closed candle detected at {latest_candle_time}. Processing...")
            
            # Apply feature engineering
            df_features = feature_engineering(df.copy())
            df_features.dropna(inplace=True)
            
            if len(df_features) < SEQUENCE_LENGTH:
                print("Not enough data after dropping NaNs to form a sequence. Waiting...")
                time.sleep(60)
                continue
                
            # Scale
            df_scaled = df_features.copy()
            df_scaled[features_list] = scaler.transform(df_features[features_list])
            
            # Get the last SEQUENCE_LENGTH rows
            seq_data = df_scaled[features_list].values[-SEQUENCE_LENGTH:]
            
            # Convert to tensor: shape (1, seq_len, num_features)
            x_tensor = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(x_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze()
                
                prob_sell = probs[0].item()
                prob_buy = probs[1].item()
                
                confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
                
                if prob_buy >= confidence_threshold:
                    pred = 1 # Buy
                elif prob_sell >= confidence_threshold:
                    pred = 0 # Sell
                else:
                    pred = -1 # Hold
                
            print(f"[{datetime.now()}] Prob Sell: {prob_sell:.2f}, Prob Buy: {prob_buy:.2f} -> Pred index: {pred}")
            
            action = "Hold"
            if pred == 0:
                action = "Sell"
            elif pred == 1:
                action = "Buy"
                
            print(f"[{datetime.now()}] Model Prediction: {action}")
            
            if not dry_run:
                # Example execution (you should adjust position sizing and risk management!)
                if action == "Buy":
                    print("Executing Market Buy Order...")
                    # exchange.create_market_buy_order(SYMBOL, 0.001) 
                elif action == "Sell":
                    print("Executing Market Sell Order...")
                    # exchange.create_market_sell_order(SYMBOL, 0.001)
                    
            last_processed_candle = latest_candle_time
            
        except Exception as e:
            print(f"[{datetime.now()}] Error in bot loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
