import os
import numpy as np
from diplomovka.config import configSpectrogram
from diplomovka.spectrograms import tensorflowSpectrograms, librosaSpectrograms
import soundfile as sf


def extract_spectrograms_from_blocks(file_path, library, spectrogram_type, overlap=configSpectrogram.overlap, frame_length=configSpectrogram.frame_length):
    # Load the FLAC audio file
    audio_data, sample_rate = sf.read(file_path)
    # Calculate block size and overlap
    block_size = int(frame_length * sample_rate)
    overlap_size = int(overlap * block_size)
    # Calculate number of frames and create overlapping blocks
    num_frames = (len(audio_data) - overlap_size) // (block_size - overlap_size)
    frames = [audio_data[i * (block_size - overlap_size): i * (block_size - overlap_size) + block_size]
              for i in range(num_frames)]
    # Calculate spectrogram for each frame
    if library == 'tf':
        # Choose spectrogram type
        if spectrogram_type == 'mel':
            spectrograms = [tensorflowSpectrograms.extract_mel_spectrogram(frame, sample_rate) for frame in frames]
        elif spectrogram_type == 'log_mel':
            spectrograms = [tensorflowSpectrograms.extract_log_mel_spectrogram(frame, sample_rate) for frame in frames]
        elif spectrogram_type == 'mfcc':
            spectrograms = [tensorflowSpectrograms.extract_mfcc(frame, sample_rate) for frame in frames]
        else:
            raise ValueError("Invalid spectrogram type. Choose from 'mel', 'log_mel', or 'mfcc'.")
    elif library == 'librosa':
        if spectrogram_type == 'mel':
            spectrograms = [librosaSpectrograms.extract_mel_spectrogram_librosa(frame, sample_rate) for frame in frames]
        elif spectrogram_type == 'log_mel':
            spectrograms = [librosaSpectrograms.extract_log_mel_spectrogram_librosa(frame, sample_rate) for frame in frames]
        elif spectrogram_type == 'mfcc':
            spectrograms = [librosaSpectrograms.extract_mfcc_librosa(frame, sample_rate) for frame in frames]
        else:
            raise ValueError("Invalid spectrogram type. Choose from 'mel', 'log_mel', or 'mfcc'.")
    else:
        raise ValueError("Invalid library type")
    # Optional: Visualize or Save Spectrograms
    # for i, spectrogram in enumerate(spectrograms):
    #     plt.imshow(librosa.power_to_db(spectrogram), aspect='auto', cmap='viridis')
    #     plt.colorbar(format='%+2.0f dB')
    #     plt.title(f'Spectrogram - Frame {i + 1}')
    #     plt.show()
    return spectrograms


def preprocess_data(data_path, output_path, library, spectrogram_type):
    for speaker_folder in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_folder)
        output_speaker_path = os.path.join(output_path, library, spectrogram_type,speaker_folder)

        os.makedirs(output_speaker_path, exist_ok=True)

        for audio_file in os.listdir(speaker_path):
            audio_file_path = os.path.join(speaker_path, audio_file)

            # Extract frames and save as numpy files
            spectrograms = extract_spectrograms_from_blocks(audio_file_path, library, spectrogram_type)
            for i, spectrogram in enumerate(spectrograms):
                output_file_path = os.path.join(output_speaker_path, f"{os.path.splitext(audio_file)[0]}_spectrogram_{i}.npy")
                np.save(output_file_path, spectrograms[i].numpy())

if __name__ == "__main__":
    #separating audio files into frames
    train_data_path = os.path.join('data','train-data1')
    test_data_path = os.path.join('data','val-data')
    train_output_path = os.path.join('data','train-data-spectrogram')
    test_output_path = os.path.join('data','val-data-spectrogram')

    #Preprocess training data
    preprocess_data(train_data_path, train_output_path, 'tf', 'mel')
    #Preprocess test data
    preprocess_data(test_data_path, test_output_path, 'tf', 'mel')
