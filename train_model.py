import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
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


def bhs_grade(mae: float) -> str:
    if mae <= 5:
        return "A"
    elif mae <= 10:
        return "B"
    elif mae <= 15:
        return "C"
    return "D (below clinical standard)"


def bhs_within(errors: np.ndarray) -> dict:
    return {
        "≤5 mmHg":  float((errors <= 5).mean()  * 100),
        "≤10 mmHg": float((errors <= 10).mean() * 100),
        "≤15 mmHg": float((errors <= 15).mean() * 100),
    }


def plot_results(preds: np.ndarray, labels: np.ndarray, out_path: str = "bp_results.png") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Team 04 — Wearable BP Estimation: Prediction Results", fontsize=14, fontweight="bold")

    for ax, idx, title, color in [
        (axes[0], 0, "Systolic BP (SBP)", "#e74c3c"),
        (axes[1], 1, "Diastolic BP (DBP)", "#2980b9"),
    ]:
        errs = np.abs(preds[:, idx] - labels[:, idx])
        mae  = errs.mean()
        grade = bhs_grade(mae)
        within = bhs_within(errs)

        ax.scatter(labels[:, idx], preds[:, idx], alpha=0.3, s=10, color=color)
        lo = min(labels[:, idx].min(), preds[:, idx].min()) - 5
        hi = max(labels[:, idx].max(), preds[:, idx].max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect fit")
        ax.set_xlabel("True BP (mmHg)")
        ax.set_ylabel("Predicted BP (mmHg)")
        ax.set_title(f"{title}\nMAE = {mae:.2f} mmHg  |  BHS Grade {grade}")
        ax.legend()

        info = "\n".join(f"{k}: {v:.1f}%" for k, v in within.items())
        ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Results plot saved → {out_path}")
    plt.close()


def train():
    print("=" * 70)
    print("Wearable-Enabled AI for Cuffless Blood Pressure Estimation")
    print("Future17 SDG Challenge 2025 | Ekak Innovations | Team 04")
    print("=" * 70)

    print("\nLoading preprocessed data...")
    data = np.load("processed_data.npz")
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    print(f"Dataset: {len(X)} windows, input shape {X.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"Split → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    y_scaler = StandardScaler()
    y_train  = y_scaler.fit_transform(y_train)
    y_val    = y_scaler.transform(y_val)
    y_test   = y_scaler.transform(y_test)
    joblib.dump(y_scaler, "y_scaler.pkl")

    def make_loader(X_, y_, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_.astype(np.float32))),
            batch_size=256, shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  shuffle=False)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = BiLSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: BiLSTM | trainable parameters: {param_count:,} | device: {device}")
    print(f"Training for {num_epochs} epochs...\n")

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                val_loss += criterion(model(bx.to(device)), by.to(device)).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:>3}/{num_epochs}]  "
                  f"Train: {avg_train:.4f}  Val: {avg_val:.4f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "bp_model_best.pth")

    print("\nEvaluating best checkpoint on held-out test set...")
    model.load_state_dict(torch.load("bp_model_best.pth", weights_only=True))
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds_all.append(model(bx.to(device)).cpu().numpy())
            labels_all.append(by.numpy())

    preds_all  = y_scaler.inverse_transform(np.concatenate(preds_all))
    labels_all = y_scaler.inverse_transform(np.concatenate(labels_all))

    sbp_errors = np.abs(preds_all[:, 0] - labels_all[:, 0])
    dbp_errors = np.abs(preds_all[:, 1] - labels_all[:, 1])
    sbp_mae = sbp_errors.mean()
    dbp_mae = dbp_errors.mean()

    print("\n" + "=" * 60)
    print("RESULTS — British Hypertension Society (BHS) Protocol")
    print("=" * 60)
    print(f"  SBP MAE : {sbp_mae:.2f} mmHg  →  BHS Grade {bhs_grade(sbp_mae)}")
    print(f"  DBP MAE : {dbp_mae:.2f} mmHg  →  BHS Grade {bhs_grade(dbp_mae)}")
    print()
    for label, errors in [("SBP", sbp_errors), ("DBP", dbp_errors)]:
        w = bhs_within(errors)
        print(f"  {label} predictions within:  "
              f"5 mmHg: {w['≤5 mmHg']:.1f}%  |  "
              f"10 mmHg: {w['≤10 mmHg']:.1f}%  |  "
              f"15 mmHg: {w['≤15 mmHg']:.1f}%")
    print("=" * 60)

    plot_results(preds_all, labels_all)
    torch.save(model.state_dict(), "bp_model.pth")
    print("\nSaved: bp_model.pth, bp_model_best.pth, y_scaler.pkl, bp_results.png")


if __name__ == "__main__":
    try:
        train()
    except Exception:
        import traceback
        traceback.print_exc()
