import scipy.io
import scipy.signal
import numpy as np

def bandpass_filter(data, lowcut=0.5, highcut=8.0, fs=125, order=4):
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return scipy.signal.filtfilt(b, a, data)

def preprocess(mat_path, output_path, num_patients=1000, max_windows=5000):
    print(f"Loading {num_patients} patients from {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    p   = mat['p'][0]

    fs          = 125
    window_size = fs * 5   # 625 samples = 5 seconds
    stride      = fs * 2   # 250 samples = 2 second hop (overlapping windows)

    X, y = [], []

    for i in range(min(num_patients, len(p))):
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

            if not (60 < sbp < 200 and 40 < dbp < 120 and sbp > dbp + 20):
                continue

            pmin, pmax = ppg_win.min(), ppg_win.max()
            if pmax <= pmin:
                continue
            ppg_win = (ppg_win - pmin) / (pmax - pmin)

            X.append(ppg_win)
            y.append([sbp, dbp])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Total valid windows: {len(X)}")

    if len(X) > max_windows:
        idx = np.random.choice(len(X), max_windows, replace=False)
        X, y = X[idx], y[idx]
        print(f"Sampled {max_windows} windows.")

    X = np.expand_dims(X, axis=-1)
    np.savez(output_path, X=X, y=y)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    preprocess(
        r'part_1.mat',
        r'processed_data.npz'
    )

