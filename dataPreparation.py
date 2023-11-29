import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_io as tfio


def split_dataset(dataset_path, train_path, test_path, split_ratio=0.9):
    # Create Train and Test Directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for speaker_folder in os.listdir(dataset_path):
        speaker_path = os.path.join(dataset_path, speaker_folder)

        # Step 1: Calculate Total Audio Length per Speaker
        total_length = sum([os.path.getsize(os.path.join(speaker_path, audio_file))
                            for audio_file in os.listdir(speaker_path)])

        # Step 2: Calculate Split Points
        split_point = int(total_length * split_ratio)

        # Step 3: Iterate through audio files and copy to train/test based on split points
        audio_files = [(audio_file, os.path.getsize(os.path.join(speaker_path, audio_file)))
                       for audio_file in os.listdir(speaker_path)]
        audio_files.sort(key=lambda x: x[1], reverse=True)

        current_length = 0
        for audio_file, file_length in audio_files:
            print(audio_file, file_length)
            audio_file_path = os.path.join(speaker_path, audio_file)
            current_length += file_length

            if current_length <= split_point:
                destination = os.path.join(train_path, speaker_folder)
            else:
                destination = os.path.join(test_path, speaker_folder)

            os.makedirs(destination, exist_ok=True)
            shutil.copy(audio_file_path, destination)

#split_dataset(os.path.join('data','dev-clean','LibriSpeech','dev-clean'), os.path.join('data','train-data'), os.path.join('data','test-data'))

#GET BYTES PER SAMPLE
# audio_file_path = os.path.join('data','dev-clean','LibriSpeech','dev-clean','422','422-122949-0000.flac')
# # Read audio file using TensorFlow I/O
# audio = tfio.audio.AudioIOTensor(audio_file_path)
# # Get the bytes per sample
# bytes_per_sample = audio.dtype.size // audio.shape[-1]
# #print(f"Bytes per sample: {bytes_per_sample}")

