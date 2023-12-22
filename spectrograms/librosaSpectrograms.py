import librosa
import numpy as np
from diplomovka.config import configSpectrogram



def extract_mel_spectrogram_librosa(frame,
                                    sample_rate=configSpectrogram.sample_rate,
                                    n_fft=configSpectrogram.spectrogram_window_length,
                                    hop_length=configSpectrogram.spectrogram_window_length/2,
                                    mels=128,
                                    fmin=0.0,
                                    fmax=configSpectrogram.sample_rate/2):
    # Ensure the frame has a floating-point dtype
    frame = frame.numpy().astype(np.float32)
    # Convert frame to Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=frame,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mels,
        fmin=fmin,
        fmax=fmax
    )
    return mel_spectrogram


def extract_log_mel_spectrogram_librosa(frame,
                                        sample_rate=configSpectrogram.sample_rate,
                                        n_fft=configSpectrogram.spectrogram_window_length,
                                        hop_length=configSpectrogram.spectrogram_window_length/2,
                                        mels=128,
                                        fmin=0.0,
                                        fmax=configSpectrogram.sample_rate/2):
    # Convert frame to Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=frame.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_spectrogram = np.log1p(mel_spectrogram)  # Log-scale for stability

    return mel_spectrogram


def extract_mfcc_librosa(frame,
                         sample_rate=configSpectrogram.sample_rate,
                         n_fft=configSpectrogram.spectrogram_window_length,
                         hop_length=configSpectrogram.spectrogram_window_length/2,
                         mels=128,
                         fmin=0.0,
                         fmax=configSpectrogram.sample_rate/2,
                         dct_coefficient_count=13):
    # Convert frame to Mel spectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(
        y=frame.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_spectrogram = np.log1p(mel_spectrogram)  # Log-scale for stability

    # Compute MFCC using librosa
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(mel_spectrogram),
        n_mfcc=dct_coefficient_count
    )

    return mfcc