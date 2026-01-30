import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_mfcc_features(audio_array):
    sample_rate = 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(audio_array, frame_length=1024, frame_step=256,fft_length=1024)
    spectrograms = tf.abs(stfts)
#     spectrograms = tf.cast(stfts + 1, tf.float32)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.cast(tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz), tf.float32)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]

cnn = tf.keras.Sequential([
    tf.keras.layers.Lambda(generate_mfcc_features, input_shape=(40000,)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=2, padding='same', activation='relu')
])