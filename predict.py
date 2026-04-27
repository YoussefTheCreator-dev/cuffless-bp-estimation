"""
predict.py — Run inference with the pre-trained BiLSTM model.

Usage:
    python predict.py                     # demo on 5 samples from processed_data.npz
    python predict.py --n_samples 20      # demo on 20 samples
"""
import argparse
import sys
import numpy as np
import torch
import joblib

from train_model import BiLSTMModel


def load_model(model_path: str = "bp_model_best.pth",
               scaler_path: str = "y_scaler.pkl") -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMModel().to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, device


def predict(signal: np.ndarray, model, scaler, device) -> dict:
    """
    Predict SBP and DBP from a single preprocessed PPG window.

    Args:
        signal: numpy array of shape (625,) or (625, 1) — normalised PPG window
    Returns:
        dict with 'sbp', 'dbp', and 'category' keys
    """
    x = np.array(signal, dtype=np.float32).reshape(1, 625, 1)
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(device)).cpu().numpy()
    sbp, dbp = scaler.inverse_transform(out)[0]

    if sbp < 120 and dbp < 80:
        category = "Normal"
    elif sbp < 130 and dbp < 80:
        category = "Elevated"
    elif sbp < 140 or dbp < 90:
        category = "Hypertension Stage 1"
    else:
        category = "Hypertension Stage 2"

    return {"sbp": float(sbp), "dbp": float(dbp), "category": category}


def main():
    parser = argparse.ArgumentParser(description="BP prediction using pre-trained BiLSTM model.")
    parser.add_argument("--model_path",  default="bp_model_best.pth", help="Path to model weights")
    parser.add_argument("--scaler_path", default="y_scaler.pkl",       help="Path to label scaler")
    parser.add_argument("--data_path",   default="processed_data.npz", help="Path to processed data (for demo)")
    parser.add_argument("--n_samples",   type=int, default=5,           help="Number of demo samples to run")
    args = parser.parse_args()

    print("=" * 60)
    print("Wearable BP Estimation — Inference Demo")
    print("Future17 SDG Challenge 2025 | Ekak Innovations | Team 04")
    print("=" * 60)

    # Load model
    try:
        model, scaler, device = load_model(args.model_path, args.scaler_path)
        print(f"Model loaded from '{args.model_path}' on {device}\n")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run 'python train_model.py' first to generate model weights.")
        sys.exit(1)

    # Load demo samples
    try:
        data = np.load(args.data_path)
        X, y_true = data["X"], data["y"]
    except FileNotFoundError:
        print(f"ERROR: '{args.data_path}' not found.")
        sys.exit(1)

    n = min(args.n_samples, len(X))
    print(f"{'Sample':<8} {'True SBP':>10} {'Pred SBP':>10} {'True DBP':>10} {'Pred DBP':>10}  {'Category'}")
    print("-" * 72)

    sbp_errors, dbp_errors = [], []
    for i in range(n):
        result = predict(X[i], model, scaler, device)
        true_sbp, true_dbp = y_true[i]
        sbp_err = abs(result["sbp"] - true_sbp)
        dbp_err = abs(result["dbp"] - true_dbp)
        sbp_errors.append(sbp_err)
        dbp_errors.append(dbp_err)
        print(f"{i+1:<8} {true_sbp:>10.1f} {result['sbp']:>10.1f} "
              f"{true_dbp:>10.1f} {result['dbp']:>10.1f}  {result['category']}")

    print("-" * 72)
    print(f"{'MAE':<8} {'':>10} {np.mean(sbp_errors):>10.2f} {'':>10} {np.mean(dbp_errors):>10.2f}  mmHg")
    print()
    print("Note: All values in mmHg. For clinical use, validate with a cuff device.")


if __name__ == "__main__":
    main()
