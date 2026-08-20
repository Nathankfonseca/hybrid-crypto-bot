import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CryptoDataset(Dataset):
    def __init__(self, csv_file, sequence_length=60):
        """
        Args:
            csv_file (string): Path to the csv file with scaled features.
            sequence_length (int): Number of candles to use as lookback window.
        """
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        
        # Features to use (ignoring timestamp, open, high, low, close, volume, label)
        self.feature_cols = [
            'log_return', 'rsi', 'macd', 'macd_signal', 'macd_diff',
            'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
            'bb_width', 'bb_percent', 'atr', 'volume_ratio'
        ]
        
        self.x = self.data[self.feature_cols].values
        self.y = self.data['label'].values
        
        # Labels are already 0 (Sell) and 1 (Buy)

    def __len__(self):
        # We can only create windows up to len - sequence_length
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get sequence of features
        x_seq = self.x[idx : idx + self.sequence_length]
        
        # The label is the target for the candle AFTER the sequence
        # or we could say it's the target associated with the last candle in the sequence
        # In our data_collection, 'label' at time t is the return from t to t+lookahead.
        # So if we use data up to time t, the label should be y[t].
        # Since sequence goes from idx to idx+sequence_length-1, the target is at idx + sequence_length - 1.
        y_target = self.y[idx + self.sequence_length - 1]
        
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_target, dtype=torch.long)
