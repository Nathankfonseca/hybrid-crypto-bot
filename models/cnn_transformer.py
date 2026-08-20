import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformer(nn.Module):
    def __init__(self, num_features, cnn_channels=32, kernel_size=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2, num_classes=2):
        super(CNNTransformer, self).__init__()
        
        # 1. 1D CNN for local feature extraction (Momentum/Noise reduction)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=cnn_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        self.act = F.gelu
        self.dropout_cnn = nn.Dropout(dropout)
        
        # 2. Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, d_model)) # Max 1000 sequence length
        
        # 3. Transformer Encoder for global dependency (Order Flow structural context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, num_features)
        
        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2)
        
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout_cnn(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.dropout_cnn(x)
        
        # Back to (batch, seq_len, channels) for Transformer
        x = x.transpose(1, 2)
        
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        x = self.transformer_encoder(x)
        
        # Get the representation of the last token in the sequence (Micro Trigger)
        out = x[:, -1, :]
        
        out = self.fc(out)
        return out
