import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from models.micro_mlp import MicroMLP
import os

def train_and_test_5m():
    print("Loading 1m data and resampling to 5m...")
    df_1m = pd.read_parquet('data/micro_1m.parquet')
    df_1m.set_index('timestamp', inplace=True)
    
    # Resample
    df = df_1m.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
    
    df.sort_values('timestamp', inplace=True)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['close'] - df['close'].shift(1)
    
    # Predict the return of the VERY NEXT 5m candle
    future_return = (df['close'].shift(-1) - df['close']) / df['close']
    df['label'] = np.where(future_return > 0, 1, 0)
    
    seq_len = 10
    features_list = []
    for i in range(seq_len):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)
        df[f'momentum_lag_{i}'] = df['momentum'].shift(i)
        features_list.append(f'log_return_lag_{i}')
        features_list.append(f'momentum_lag_{i}')
        
    df.dropna(inplace=True)
    
    # Last 60 days = 60 * 24 * 12 = 17280 bars
    oos_bars = 17280
    df_train = df.iloc[:-oos_bars].copy()
    df_test = df.iloc[-oos_bars:].copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features_list])
    y_train = df_train['label'].values
    
    X_test = scaler.transform(df_test[features_list])
    y_test = df_test['label'].values
    
    X_train_ts = torch.tensor(X_train, dtype=torch.float32)
    y_train_ts = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_ts, y_train_ts), batch_size=512, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training 5m MLP...")
    model.train()
    for epoch in range(15):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    print("\nEvaluating on OOS (Last 60 Days)...")
    model.eval()
    X_test_ts = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(X_test_ts)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
    acc = np.mean(preds == y_test)
    print(f"5m MLP Accuracy: {acc*100:.2f}%")
    
    # Calculate PnL with Fees
    fee_rate = 0.001
    slippage = 0.0005
    friction = fee_rate + slippage
    
    ret_true = (df_test['close'].shift(-1) - df_test['close']) / df_test['close']
    ret_true = ret_true.fillna(0).values
    
    pnl = 0.0
    trades = 0
    for i in range(len(preds)):
        if preds[i] == 1: # Buy signal
            pnl += ret_true[i] - friction
            trades += 1
            
    print(f"Total Trades in 60 days: {trades}")
    print(f"Gross PnL (Before fees): {np.sum(np.where(preds == 1, ret_true, 0))*100:.2f}%")
    print(f"Net PnL (After {friction*100}% friction): {pnl*100:.2f}%")

if __name__ == '__main__':
    train_and_test_5m()
