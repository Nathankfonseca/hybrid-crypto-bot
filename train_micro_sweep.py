import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import json

def fetch_synthetic_volume_bars(target_volume=5.0):
    print("Loading 1m data...")
    df = pd.read_parquet('data/micro_1m.parquet')
    vol_bars = []
    current_vol = 0
    if len(df) == 0: return pd.DataFrame()
    bar_open = df['open'].iloc[0]
    bar_high = df['high'].iloc[0]
    bar_low = df['low'].iloc[0]
    
    # Fast iteration
    for row in df.itertuples():
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
            
    df_vol = pd.DataFrame(vol_bars)
    print(f"Generated {len(df_vol)} volume bars.")
    return df_vol

def create_features_and_labels():
    df = fetch_synthetic_volume_bars(target_volume=5.0)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['close'] - df['close'].shift(1)
    df.dropna(inplace=True)
    
    # Label: future return over next 3 bars
    future_return = (df['close'].shift(-3) - df['close']) / df['close']
    df['label'] = np.where(future_return > 0, 1, 0)
    df['future_return'] = future_return
    
    seq_len = 10
    # Lag features for tabular models
    for i in range(seq_len):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)
        df[f'momentum_lag_{i}'] = df['momentum'].shift(i)
        
    df.dropna(inplace=True)
    return df

def evaluate(name, probs, y_true, ret_true):
    preds = (probs > 0.5).astype(int)
    acc = np.mean(preds == y_true)
    std = np.std(probs)
    
    # Naive PnL simulation (buying only)
    trade_returns = np.where(preds == 1, ret_true, 0)
    pnl = np.sum(trade_returns) * 100 # In percentage
    
    # Random Baseline
    random_preds = np.random.randint(0, 2, size=len(y_true))
    random_acc = np.mean(random_preds == y_true)
    random_trade_returns = np.where(random_preds == 1, ret_true, 0)
    random_pnl = np.sum(random_trade_returns) * 100
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc*100:.2f}% (Random: {random_acc*100:.2f}%)")
    print(f"Prob StdDev: {std:.6f} (If very close to 0 -> Mode Collapse)")
    print(f"PnL: {pnl:.2f}% (Random PnL: {random_pnl:.2f}%)")
    
    return {
        "Model": name,
        "Accuracy": acc,
        "Prob_StdDev": std,
        "PnL": pnl,
        "Random_Accuracy": random_acc,
        "Random_PnL": random_pnl
    }

def train_sweep():
    df = create_features_and_labels()
    
    # Out of Sample: Last 60 days
    # Estimating 1200 volume bars per day -> ~72,000 bars
    drop_rows = 72000
    if len(df) <= drop_rows:
        print("Not enough data to drop 60 days. Taking last 20%.")
        drop_rows = int(len(df) * 0.2)
        
    df_train = df.iloc[:-drop_rows].copy()
    df_test = df.iloc[-drop_rows:].copy()
    
    print(f"Training rows: {len(df_train)} | Testing (OOS) rows: {len(df_test)}")
    
    seq_len = 10
    features_list = []
    for i in range(seq_len):
        features_list.append(f'log_return_lag_{i}')
        features_list.append(f'momentum_lag_{i}')
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features_list])
    y_train = df_train['label'].values
    X_test = scaler.transform(df_test[features_list])
    y_test = df_test['label'].values
    ret_test = df_test['future_return'].values
    
    results = []
    
    # 1. Logistic Regression
    print("\n[1/4] Training Logistic Regression...")
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X_train, y_train)
    probs_lr = clf_lr.predict_proba(X_test)[:, 1]
    results.append(evaluate("Logistic Regression", probs_lr, y_test, ret_test))
    
    # 2. LightGBM
    print("\n[2/4] Training LightGBM...")
    clf_lgb = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.01, random_state=42)
    clf_lgb.fit(X_train, y_train)
    probs_lgb = clf_lgb.predict_proba(X_test)[:, 1]
    results.append(evaluate("LightGBM", probs_lgb, y_test, ret_test))
    
    # Prepare PyTorch Tensors
    X_train_ts = torch.tensor(X_train, dtype=torch.float32)
    y_train_ts = torch.tensor(y_train, dtype=torch.long)
    X_test_ts = torch.tensor(X_test, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_ts, y_train_ts), batch_size=512, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for NN: {device}")
    
    # 3. Simple MLP
    print("\n[3/4] Training Simple MLP...")
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.6), # Agressive dropout
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.6),
                nn.Linear(32, 2)
            )
        def forward(self, x): return self.net(x)
        
    mlp = MLP().to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=0.001, weight_decay=1e-3) # Heavy weight decay
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10): # Short epochs for baseline
        mlp.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = mlp(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    mlp.eval()
    with torch.no_grad():
        out_mlp = mlp(X_test_ts.to(device))
        probs_mlp = torch.softmax(out_mlp, dim=1)[:, 1].cpu().numpy()
    results.append(evaluate("Simple MLP", probs_mlp, y_test, ret_test))
    
    # 4. 1D CNN
    print("\n[4/4] Training 1D CNN...")
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(2, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Flatten(),
                nn.Linear(32 * 10, 2)
            )
        def forward(self, x):
            return self.conv(x)
            
    # Reshape for CNN: (batch, channels, seq_len)
    X_train_cnn = X_train_ts.view(-1, 10, 2).transpose(1, 2)
    X_test_cnn = X_test_ts.view(-1, 10, 2).transpose(1, 2)
    train_loader_cnn = DataLoader(TensorDataset(X_train_cnn, y_train_ts), batch_size=512, shuffle=True)
    
    cnn = SimpleCNN().to(device)
    optimizer_cnn = optim.AdamW(cnn.parameters(), lr=0.001, weight_decay=1e-3)
    for epoch in range(10):
        cnn.train()
        for bx, by in train_loader_cnn:
            bx, by = bx.to(device), by.to(device)
            optimizer_cnn.zero_grad()
            out = cnn(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer_cnn.step()
            
    cnn.eval()
    with torch.no_grad():
        out_cnn = cnn(X_test_cnn.to(device))
        probs_cnn = torch.softmax(out_cnn, dim=1)[:, 1].cpu().numpy()
    results.append(evaluate("Simple 1D CNN", probs_cnn, y_test, ret_test))
    
    with open('micro_1m_tournament_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nFinished! Results saved to micro_1m_tournament_results.json")

if __name__ == '__main__':
    train_sweep()
