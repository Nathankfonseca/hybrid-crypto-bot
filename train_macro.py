import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

from models.autoformer import Autoformer
from pipelines.macro_data import fetch_macro_data, apply_macro_features, generate_macro_labels

import pandas as pd
import numpy as np

def train(symbol='BTCUSDT'):
    print(f"Iniciando treinamento Macro (Autoformer) com Deep Data para {symbol}...")
    try:
        df = pd.read_parquet(f'data/macro_1h_{symbol}.parquet')
        print(f"Dados carregados do disco: {len(df)} velas.")
    except Exception as e:
        print(f"Falha ao ler parquet: {e}. Certifique-se de que o download concluiu.")
        return
        
    df = apply_macro_features(df)
    print(f"Dados brutos originais do disco: {len(df)} velas.")
    
    # Train/Test Split (Out-Of-Sample Protection)
    # Cortamos os últimos 65 dias (60 de teste + 5 de folga/indicadores) para evitar Data Leakage
    drop_rows = 24 * 65 
    df = df.iloc[:-drop_rows]
    
    print(f"Dados reservados para treinamento (OOS aplicável): {len(df)} velas.")
    
    df = generate_macro_labels(df)
    
    features_list = [
        'log_return', 'atr_14', 'atr_50', 'macd', 'macd_signal', 
        'dist_sma_50', 'dist_sma_200'
    ]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features_list])
    
    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, f'models/macro_scaler_{symbol}.pkl')
    
    # Sequence creation
    seq_len = 96 # 4 days of hourly data context
    X, y = [], []
    for i in range(len(X_scaled) - seq_len):
        X.append(X_scaled[i:i+seq_len])
        y.append(df['label'].iloc[i+seq_len])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    # Use larger batch size for massive data
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoformer(num_features=len(features_list), num_classes=2).to(device)
    
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
        print(f"Macro Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'models/macro_autoformer_{symbol}.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break
                
    print(f"Treinamento Macro concluído. Melhor modelo salvo em models/macro_autoformer_{symbol}.pt")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        train('BTCUSDT')
