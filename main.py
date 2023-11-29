import os

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

def extract_frames(audio_file, frame_duration=0.03, overlap=0.5, sample_rate=16000):
    # Read audio file using TensorFlow I/O
    audio = tfio.audio.AudioIOTensor(audio_file)

    frame_length = int(frame_duration * sample_rate)
    frame_step = int(frame_length * (1 - overlap))

    # Calculate the number of frames
    num_frames = int((len(audio) - frame_length) / frame_step) + 1
    print(audio_file, "have", num_frames, "frames")

    frames = []
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length

        # Extract frame
        frame = audio[start:end]

        frames.append(frame)

    return frames


def preprocess_data(data_path, output_path, frame_duration=0.03, overlap=0.5, sample_rate=16000):
    for speaker_folder in os.listdir(data_path):
        speaker_path = os.path.join(data_path, speaker_folder)
        output_speaker_path = os.path.join(output_path, speaker_folder)

        os.makedirs(output_speaker_path, exist_ok=True)

        for audio_file in os.listdir(speaker_path):
            audio_file_path = os.path.join(speaker_path, audio_file)

            # Extract frames and save as numpy files
            frames = extract_frames(audio_file_path, frame_duration, overlap, sample_rate)
            for i, frame in enumerate(frames):
                output_file_path = os.path.join(output_speaker_path, f"{os.path.splitext(audio_file)[0]}_frame_{i}.npy")
                np.save(output_file_path, frame.numpy())


def extract_mel_spectrogram(frame, sample_rate=16000):
    # Convert frame to Mel spectrogram
    frame = tf.cast(frame, tf.float32)
    mel_spectrogram = tfio.audio.spectrogram(frame, nfft=2048, window=256, stride=128)
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram, rate=sample_rate, mels=128, fmax=8000, fmin=0)

    return mel_spectrogram


def extract_log_mel_spectrogram(frame, sample_rate=16000):
    # Convert frame to Mel spectrogram
    mel_spectrogram = tfio.audio.spectrogram(frame, nfft=2048, window=256, stride=128)
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram, rate=sample_rate, mels=128, fmax=8000, fmin=0)

    # Apply logarithmic scaling
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return log_mel_spectrogram


def extract_mfcc(frame, sample_rate=16000):
    # Convert frame to Mel spectrogram
    mel_spectrogram = tfio.audio.spectrogram(frame, nfft=2048, window=256, stride=128)
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram, rate=sample_rate, mels=128, fmax=8000, fmin=0)

    # Convert Mel spectrogram to MFCC
    mfcc = tfio.audio.mfcc(mel_spectrogram, rate=sample_rate, dct_coefficient_count=13)

    return mfcc


def generate_spectrograms_from_frames(input_path, output_path, spectrogram_type='mel'):
    for speaker_folder in os.listdir(input_path):
        speaker_path = os.path.join(input_path, speaker_folder)
        output_speaker_path = os.path.join(output_path, speaker_folder)

        os.makedirs(output_speaker_path, exist_ok=True)

        for frame_file in os.listdir(speaker_path):
            frame_file_path = os.path.join(speaker_path, frame_file)

            # Load frame
            frame = np.load(frame_file_path)

            # Choose spectrogram type
            if spectrogram_type == 'mel':
                spectrogram = extract_mel_spectrogram(frame)
            elif spectrogram_type == 'log_mel':
                spectrogram = extract_log_mel_spectrogram(frame)
            elif spectrogram_type == 'mfcc':
                spectrogram = extract_mfcc(frame)
            else:
                raise ValueError("Invalid spectrogram type. Choose from 'mel', 'log_mel', or 'mfcc'.")

            # Save spectrogram as numpy file
            output_file_path = os.path.join(output_speaker_path, f"{os.path.splitext(frame_file)[0]}_{spectrogram_type}_spectrogram.npy")
            np.save(output_file_path, spectrogram.numpy())

if __name__ == "__main__":
    #separating audio files into frames
    # train_data_path = os.path.join('data','train-data')
    # test_data_path = os.path.join('data','test-data')
    # train_output_path = os.path.join('data','train-data-processed-frames')
    # test_output_path = os.path.join('data','test-data-processed-frames')

    # Preprocess training data
    #preprocess_data(train_data_path, train_output_path)
    # Preprocess test data
    #preprocess_data(test_data_path, test_output_path)

    train_frames_path = os.path.join('data','train-data-processed-frames')
    test_frames_path = os.path.join('data','test-data-processed-frames')

    train_spectrograms_path = os.path.join('data','spectrograms','mel-train')
    test_spectrograms_path = os.path.join('data','spectrograms','mel-test')

    # Generate Mel spectrograms from training frames
    generate_spectrograms_from_frames(train_frames_path, train_spectrograms_path, spectrogram_type='mel')

    # Generate Mel spectrograms from testing frames
    generate_spectrograms_from_frames(test_frames_path, test_spectrograms_path, spectrogram_type='mel')

