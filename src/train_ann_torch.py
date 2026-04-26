import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc

# Paths
DATA_DIR = os.path.abspath('data')
TRAIN_CSV = os.path.join(DATA_DIR, 'train_balanced.csv')
TEST_CSV  = os.path.join(DATA_DIR, 'test.csv')

class SimpleANN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

def train_ann_torch():
    print("--- Starting PyTorch ANN Training ---")
    
    # Check for MPS (Mac GPU)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading datasets...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)

    X_train = train.drop(columns=['isFraud']).values.astype(np.float32)
    y_train = train['isFraud'].values.reshape(-1, 1).astype(np.float32)
    X_test  = test.drop(columns=['isFraud', 'TransactionID'], errors='ignore').values.astype(np.float32)
    y_test  = test['isFraud'].values.astype(np.float32)

    print("Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # To Tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t  = torch.from_numpy(X_test).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    model = SimpleANN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training (20 epochs for speed)...")
    model.train()
    for epoch in range(20):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/20, Loss: {running_loss/len(train_loader):.4f}")

    print("\nEvaluating...")
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_t).cpu().numpy().flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")
    print("--- PyTorch Training Complete ---")

if __name__ == "__main__":
    train_ann_torch()
