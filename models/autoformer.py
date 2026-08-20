import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism using FFT
    """
    def __init__(self, d_model, n_heads):
        super(AutoCorrelation, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)

        # Fast Fourier Transform for Auto-Correlation
        q_fft = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)
        
        # Simple attention-like weighting using Top-k autocorrelations (simplified)
        weights = F.softmax(corr, dim=-1)
        
        # Re-arrange and multiply by values
        out = torch.einsum('b h d l, b h d l -> b h l d', weights, v.permute(0, 2, 3, 1)).contiguous()
        out = out.view(B, L, -1)
        
        return self.out_projection(out)

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, moving_avg=25, dropout=0.1):
        super(AutoformerEncoderLayer, self).__init__()
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.autocorrelation = AutoCorrelation(d_model, n_heads)
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. Auto-Correlation
        new_x = self.autocorrelation(x, x, x)
        
        # 2. Add & Norm (Decomp)
        x = x + self.dropout(new_x)
        x, trend1 = self.decomp1(x)
        
        # 3. Feed Forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 4. Add & Norm (Decomp)
        res, trend2 = self.decomp2(x + y)
        
        return res, trend1 + trend2

class Autoformer(nn.Module):
    def __init__(self, num_features, d_model=64, n_heads=4, e_layers=2, d_ff=256, moving_avg=25, dropout=0.1, num_classes=3):
        super(Autoformer, self).__init__()
        self.enc_embedding = nn.Linear(num_features, d_model)
        
        self.encoder_layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg, dropout)
            for _ in range(e_layers)
        ])
        
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model, num_classes)
        
    def forward(self, x_enc):
        x = self.enc_embedding(x_enc)
        
        trends = []
        for layer in self.encoder_layers:
            x, trend = layer(x)
            trends.append(trend)
            
        # We use the final state to predict direction (Bullish, Bearish, Neutral)
        # Pulling the last timestep representation
        out = x[:, -1, :] 
        out = self.dropout(self.act(out))
        out = self.projection(out)
        return out
