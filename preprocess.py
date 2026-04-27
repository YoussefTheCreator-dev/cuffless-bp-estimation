import argparse
import sys
import scipy.io
import scipy.signal
import numpy as np


def bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 8.0,
                    fs: int = 125, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return scipy.signal.filtfilt(b, a, data)


def preprocess(mat_path: str, output_path: str,
               num_patients: int = 1000, max_windows: int = 5000) -> None:
    print(f"Loading {num_patients} patients from {mat_path}...")
    try:
        mat = scipy.io.loadmat(mat_path)
    except FileNotFoundError:
        print(f"ERROR: '{mat_path}' not found.")
        print("Download part_1.mat from: https://www.kaggle.com/datasets/mkachuee/cuff-less-blood-pressure-estimation")
        sys.exit(1)

    p = mat['p'][0]

    fs          = 125
    window_size = fs * 5   # 625 samples = 5 seconds
    stride      = fs * 2   # 250 samples = 2-second hop

    X, y = [], []

    for i in range(min(num_patients, len(p))):
        try:
            ppg_raw  = p[i][0].flatten()
            abp_raw  = p[i][1].flatten()
            ppg_filt = bandpass_filter(ppg_raw, fs=fs)

            for start in range(0, len(ppg_filt) - window_size, stride):
                end     = start + window_size
                ppg_win = ppg_filt[start:end].copy()
                abp_win = abp_raw[start:end]

                # Percentile labels — robust against noise spikes
                sbp = float(np.percentile(abp_win, 95))
                dbp = float(np.percentile(abp_win,  5))

                # Reject physiologically implausible windows
                if not (60 < sbp < 200 and 40 < dbp < 120 and sbp > dbp + 20):
                    continue

                pmin, pmax = ppg_win.min(), ppg_win.max()
                if pmax <= pmin:
                    continue
                ppg_win = (ppg_win - pmin) / (pmax - pmin)

                X.append(ppg_win)
                y.append([sbp, dbp])

        except Exception as e:
            print(f"  Skipping patient {i}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{min(num_patients, len(p))} patients, "
                  f"{len(X)} valid windows so far...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Total valid windows: {len(X)}")

    if len(X) == 0:
        print("ERROR: No valid windows extracted. Check the input file.")
        sys.exit(1)

    if len(X) > max_windows:
        idx = np.random.choice(len(X), max_windows, replace=False)
        X, y = X[idx], y[idx]
        print(f"Randomly sampled {max_windows} windows.")

    X = np.expand_dims(X, axis=-1)   # (N, 625, 1) for LSTM input
    np.savez(output_path, X=X, y=y)
    print(f"Saved → {output_path}  (X: {X.shape}, y: {y.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess raw .mat PPG data for BiLSTM BP estimation.")
    parser.add_argument("--mat_path",    default="part_1.mat",        help="Path to raw .mat file")
    parser.add_argument("--output_path", default="processed_data.npz", help="Output .npz path")
    parser.add_argument("--num_patients", type=int, default=1000,      help="Max patients to process")
    parser.add_argument("--max_windows",  type=int, default=5000,      help="Max windows to keep")
    args = parser.parse_args()

    preprocess(args.mat_path, args.output_path, args.num_patients, args.max_windows)
