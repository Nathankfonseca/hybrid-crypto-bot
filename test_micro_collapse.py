import torch
import numpy as np
from models.cnn_transformer import CNNTransformer

def test():
    device = torch.device('cpu')
    model = CNNTransformer(num_features=2, num_classes=2).to(device)
    model.load_state_dict(torch.load('models/micro_cnn.pt', map_location=device))
    model.eval()

    print("Testing micro_model with random inputs to check if it's collapsed:")
    for i in range(5):
        # Generate random input of shape (1, 10, 2)
        x = torch.randn(1, 10, 2)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1).squeeze().numpy()
        print(f"Random Input {i+1} -> BUY {probs[1]*100:.6f}% | SELL {probs[0]*100:.6f}%")

if __name__ == '__main__':
    test()
