import os

import numpy as np
import matplotlib.pyplot as plt

# window_size = 256
# window = np.hanning(window_size)
#
# plt.plot(window)
# plt.title('Hann Window')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.show()
#
# # Signal
# signal = np.sin(np.linspace(0, 6 * np.pi, 1000))
#
# # Window size and stride
# window_size = 100
# stride = 50
#
# # Plot signal
# plt.plot(signal, label='Signal')
#
# # Plot windows with overlap
# for i in range(0, len(signal) - window_size, stride):
#     plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
#
# plt.title('Signal with Overlapping Windows (Stride=50)')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

def calculate_total_duration_seconds(speaker_path, sampling_rate=16000, bytes_per_sample=2, num_channels=1):
    total_size_bytes = sum([os.path.getsize(os.path.join(speaker_path, audio_file))
                            for audio_file in os.listdir(speaker_path)])
    total_duration_seconds = total_size_bytes / (bytes_per_sample * num_channels * sampling_rate)
    return total_duration_seconds


def generate_histogram(dataset_path):
    speakers = os.listdir(dataset_path)
    total_durations = [calculate_total_duration_seconds(os.path.join(dataset_path, speaker)) for speaker in speakers]

    # Plotting the Histogram
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis.colors  # You can choose a different colormap

    x = np.arange(len(speakers))
    plt.bar(x, total_durations, color=colors, alpha=0.7)
    plt.xticks(x, speakers, rotation=45, ha='right')

    plt.xlabel('Speakers')
    plt.ylabel('Total Audio Duration (seconds)')
    plt.title('Distribution of Audio Durations for Each Speaker')

    plt.tight_layout()
    plt.show()


generate_histogram(os.path.join('data','dev-clean','LibriSpeech','dev-clean'))