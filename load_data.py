import argparse
import scipy.io
import matplotlib.pyplot as plt


def load_and_plot(mat_path: str, output_image_path: str = "raw_data_check.png") -> None:
    """
    Loads a .mat file containing PPG, ABP, and ECG signals, prints basic metadata,
    and plots the first 500 samples of the first patient.
    
    Args:
        mat_path (str): Path to the .mat dataset.
        output_image_path (str): Path to save the resulting plot image.
    """
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
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 1, 2)
    plt.plot(abp[:500], color='orange')
    plt.title("ABP Signal (Blood Pressure)")
    plt.ylabel("mmHg")
    
    plt.subplot(3, 1, 3)
    plt.plot(ecg[:500], color='green')
    plt.title("ECG Signal")
    plt.ylabel("Amplitude")
    plt.xlabel("Samples (at 125Hz)")
    
    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Plot saved as {output_image_path} in the project folder.")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and plot raw physiological signals from .mat file.")
    parser.add_argument("--mat_path", type=str, default="part_1.mat", help="Path to the raw .mat data file.")
    parser.add_argument("--output_image", type=str, default="raw_data_check.png", help="Path to save the generated plot.")
    args = parser.parse_args()
    
    load_and_plot(args.mat_path, args.output_image)
