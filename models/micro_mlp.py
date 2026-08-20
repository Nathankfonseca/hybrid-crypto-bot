import torch.nn as nn

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
        
    def forward(self, x):
        return self.net(x)
