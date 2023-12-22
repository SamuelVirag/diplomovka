import numpy as np
import matplotlib.pyplot as plt

def plot_spectrogram_from_npy(file_path):
    # Load the spectrogram from .npy file
    spectrogram = np.load(file_path)

    # Plot the spectrogram
    plt.imshow(spectrogram, aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    #plt.title(f'Spectrogram - Frame {i + 1}')
    plt.show()

if __name__ == "__main__":
    # Specify the path to the .npy file
    npy_file_path = "/diplomovka/data/train-data-spectrogram/tf/mel/174/174-50561-0000_spectrogram_0.npy"

    # Plot the spectrogram
    plot_spectrogram_from_npy(npy_file_path)

