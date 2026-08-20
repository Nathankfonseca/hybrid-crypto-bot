import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from dataset import CryptoDataset
from model import TimeSeriesTransformer
import os

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
SEQUENCE_LENGTH = 60
NUM_FEATURES = 12 # Based on data_collection.py

def train_model():
    print("Loading dataset...")
    dataset = CryptoDataset('dataset.csv', sequence_length=SEQUENCE_LENGTH)
    
    # Train / Val Split (Chronological to avoid lookahead bias)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TimeSeriesTransformer(num_features=NUM_FEATURES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training loop...")
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
            
        train_loss = train_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
                
        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("  --> Saved new best model")

    print("Training complete!")

if __name__ == "__main__":
    train_model()
