import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import ccxt
import pandas as pd
import numpy as np

from models.cnn_transformer import CNNTransformer

def fetch_synthetic_volume_bars(target_volume=5.0):
    """ Read 1m data from parquet and group it to simulate volume bars """
    df = pd.read_parquet('data/micro_1m.parquet')
    print(f"Dados brutos carregados: {len(df)} velas de 1m.")
    
    # We will use itertuples for speed over 1M rows
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
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': row.close,
                'volume': current_vol
            })
            current_vol = 0
            # Approx open for next
            bar_open = row.close
            bar_high = row.close
            bar_low = row.close
                
    df_vol = pd.DataFrame(vol_bars)
    print(f"Volume bars geradas: {len(df_vol)}")
    return df_vol

def train():
    print("Iniciando treinamento Micro (CNN-Transformer) com Deep Data...")
    df = fetch_synthetic_volume_bars(target_volume=5.0)
    print(f"Volume bars geradas: {len(df)}")
    
    # Train/Test Split (Out-Of-Sample Protection)
    # 60 dias (estimando ~1200 barras de volume por dia) -> Cortando as últimas 72.000 barras
    drop_rows = 60 * 1200
    if len(df) > drop_rows:
        df = df.iloc[:-drop_rows]
    print(f"Volume bars para treinamento (OOS aplicável): {len(df)}")
    
    # Feature Engineering
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['momentum'] = df['close'] - df['close'].shift(1)
    df.dropna(inplace=True)
    
    # Labels: Perfeitamente balanceadas 50/50 usando a mediana
    future_return = (df['close'].shift(-3) - df['close']) / df['close']
    median_ret = future_return.median()
    df['label'] = np.where(future_return > median_ret, 1, 0)
    df.dropna(inplace=True)
    
    features_list = ['log_return', 'momentum']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_list])
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/micro_scaler.pkl')
    
    seq_len = 10 # Context of 10 volume bars
    X, y = [], []
    for i in range(len(X_scaled) - seq_len):
        X.append(X_scaled[i:i+seq_len])
        y.append(df['label'].iloc[i+seq_len])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True) # Larger batch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNTransformer(num_features=len(features_list), num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        print(f"Micro Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'models/micro_cnn.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
                
    print("Treinamento Micro concluído. Melhor modelo salvo em models/micro_cnn.pt")

if __name__ == "__main__":
    train()
