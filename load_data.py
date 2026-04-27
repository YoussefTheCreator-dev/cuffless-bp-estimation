import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def load_and_plot(mat_path):
    print(f"Loading data from {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    
    # Print keys and structure
    print(f"Keys in the .mat file: {mat.keys()}")
    p = mat['p'][0]
    print(f"Number of patients: {len(p)}")
    
    # Get the first patient's data
    patient_1 = p[0]
    print(f"Signal shape (signals, samples): {patient_1.shape}")
    
    ppg = patient_1[0]
    abp = patient_1[1]
    ecg = patient_1[2]
    
    # Plot first 500 samples (4 seconds at 125Hz)
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    plt.plot(ppg[:500])
    plt.title("Raw PPG Signal")
    
    plt.subplot(3, 1, 2)
    plt.plot(abp[:500])
    plt.title("ABP Signal (Blood Pressure)")
    
    plt.subplot(3, 1, 3)
    plt.plot(ecg[:500])
    plt.title("ECG Signal")
    
    plt.tight_layout()
    plt.savefig(r'raw_data_check.png')
    print("Plot saved as raw_data_check.png in the project folder.")
    plt.show()

if __name__ == "__main__":
    load_and_plot(r'part_1.mat')

