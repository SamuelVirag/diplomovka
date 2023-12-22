import tensorflow as tf
import tensorflow_io as tfio
from diplomovka.config import configSpectrogram


def extract_mel_spectrogram(frame, sample_rate=configSpectrogram.sample_rate):
    # Convert frame to Mel spectrogram
    frame = tf.cast(frame, tf.float32)
    mel_spectrogram = tfio.audio.spectrogram(frame,
                                             nfft=configSpectrogram.spectrogram_window_length,#480
                                             window=configSpectrogram.spectrogram_window_length, #480
                                             stride=tf.cast(configSpectrogram.spectrogram_window_length/2, dtype=tf.int32)) #240
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram,
                                          rate=sample_rate, #16000
                                          mels=128,
                                          fmax=configSpectrogram.sample_rate/2, #8000
                                          fmin=0)
    return mel_spectrogram


def extract_log_mel_spectrogram(frame, sample_rate=configSpectrogram.sample_rate):
    # Convert frame to Mel spectrogram
    mel_spectrogram = tfio.audio.spectrogram(frame,
                                             nfft=configSpectrogram.spectrogram_window_length,
                                             window=configSpectrogram.spectrogram_window_length,
                                             stride=tf.cast(configSpectrogram.spectrogram_window_length/2, dtype=tf.int32))
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram,
                                          rate=sample_rate,
                                          mels=128,
                                          fmax=configSpectrogram.sample_rate/2,
                                          fmin=0)

    # Apply logarithmic scaling
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return log_mel_spectrogram


def extract_mfcc(frame, sample_rate=configSpectrogram.sample_rate):
    # Convert frame to Mel spectrogram
    mel_spectrogram = tfio.audio.spectrogram(frame,
                                             nfft=configSpectrogram.spectrogram_window_length,
                                             window=configSpectrogram.spectrogram_window_length,
                                             stride=tf.cast(configSpectrogram.spectrogram_window_length/2, dtype=tf.int32))
    mel_spectrogram = tfio.audio.melscale(mel_spectrogram,
                                          rate=sample_rate,
                                          mels=128,
                                          fmax=configSpectrogram.sample_rate/2,
                                          fmin=0)

    # Convert Mel spectrogram to MFCC
    mfcc = tfio.audio.mfcc(mel_spectrogram, rate=sample_rate, dct_coefficient_count=13)

    return mfcc