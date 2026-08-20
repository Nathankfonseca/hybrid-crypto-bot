import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1, num_classes=2):
        super(TimeSeriesTransformer, self).__init__()
        
        # 1. Project input features to d_model dimensions
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 2. Add Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final Classification Head
        # We will use the output of the last time step for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        
        # Project
        x = self.input_projection(x) # (batch_size, seq_len, d_model)
        
        # Add position info
        x = self.positional_encoding(x)
        
        # Transform
        out = self.transformer_encoder(x) # (batch_size, seq_len, d_model)
        
        # Take the output of the last time step in the sequence
        last_step_out = out[:, -1, :] # (batch_size, d_model)
        
        # Classify
        logits = self.fc(last_step_out) # (batch_size, num_classes)
        return logits
