import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class MicroMLP(nn.Module):
    def __init__(self, num_features=20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

def fetch_synthetic_volume_bars(target_volume=50.0):
    df = pd.read_parquet('data/micro_1m.parquet')
    vol_bars = []
    current_vol = 0
    if len(df) == 0: return pd.DataFrame()
    bar_open = df['open'].iloc[0]
    bar_high = df['high'].iloc[0]
    bar_low = df['low'].iloc[0]
    
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
    return pd.DataFrame(vol_bars)

def train_and_save():
    df = fetch_synthetic_volume_bars(target_volume=50.0)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['close'] - df['close'].shift(1)
    df.dropna(inplace=True)
    
    future_return = (df['close'].shift(-3) - df['close']) / df['close']
    df['label'] = np.where(future_return > 0, 1, 0)
    
    seq_len = 10
    features_list = []
    for i in range(seq_len):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)
        df[f'momentum_lag_{i}'] = df['momentum'].shift(i)
        features_list.append(f'log_return_lag_{i}')
        features_list.append(f'momentum_lag_{i}')
        
    df.dropna(inplace=True)
    
    # 7200 bars for OOS testing later (last 60 days at 50 BTC per bar), so we train on the rest
    drop_rows = 7200
    df_train = df.iloc[:-drop_rows].copy()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features_list])
    y_train = df_train['label'].values
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/micro_mlp_scaler.pkl')
    
    X_train_ts = torch.tensor(X_train, dtype=torch.float32)
    y_train_ts = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_ts, y_train_ts), batch_size=512, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Micro MLP...")
    model.train()
    for epoch in range(10):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), 'models/micro_mlp.pt')
    print("Micro MLP and scaler saved successfully!")

if __name__ == '__main__':
    train_and_save()
