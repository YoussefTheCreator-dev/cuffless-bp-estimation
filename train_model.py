import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train():
    print("Loading preprocessed data...")
    data = np.load(r'processed_data.npz')
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.float32)

    # 3-way split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize labels
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)
    joblib.dump(y_scaler, r'y_scaler.pkl')

    def make_loader(X_, y_, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_.astype(np.float32))),
            batch_size=256, shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = BiLSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float('inf')
    print(f"Training on {device} for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Train: {avg_train:.4f}, Val: {avg_val:.4f}, LR: {lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), r'bp_model_best.pth')

    # Evaluate on held-out test set using best checkpoint
    model.load_state_dict(torch.load(r'bp_model_best.pth'))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds_all.append(model(bx).cpu().numpy())
            labels_all.append(by.numpy())

    preds_all  = y_scaler.inverse_transform(np.concatenate(preds_all))
    labels_all = y_scaler.inverse_transform(np.concatenate(labels_all))

    sbp_mae = np.abs(preds_all[:, 0] - labels_all[:, 0]).mean()
    dbp_mae = np.abs(preds_all[:, 1] - labels_all[:, 1]).mean()
    print(f"\nFinal Test MAE - SBP: {sbp_mae:.2f} mmHg, DBP: {dbp_mae:.2f} mmHg")

    torch.save(model.state_dict(), r'bp_model.pth')
    print("Model saved as bp_model.pth (best checkpoint)")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()

