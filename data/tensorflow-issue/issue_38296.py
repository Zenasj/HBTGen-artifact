from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


class Spectrogram(tf.keras.layers.Layer):
    def __init__(self, num_freqs, max_freq, **kwargs):
        super(Spectrogram, self).__init__(**kwargs)

        self.num_freqs = num_freqs
        self.max_freq = max_freq

        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=2), tf.keras.layers.InputSpec(ndim=2)
        ]

    def call(self, x_fs):
        x, fs = x_fs
        nfft = tf.cast(
            fs[0,0] * (self.num_freqs - 1) / self.max_freq,
            tf.int32
        )
        y = tf.signal.stft(x, nfft, 256, nfft, pad_end=True)
        y = tf.sqrt(tf.abs(y))[:, :, :self.num_freqs]
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, self.num_freqs)


signal = tf.keras.layers.Input(shape=(None,))
fs = tf.keras.layers.Input(shape=(1,))
x = Spectrogram(257, 10_000)([signal, fs])
y = tf.keras.layers.Conv1D(16, 3)(x)

tf.keras.models.Model([signal, fs], [y]).summary()