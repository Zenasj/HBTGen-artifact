# tf.random.uniform((B, None)) ← Input is a batch of 1D signals with variable length; second input is (B, 1) for sampling freq

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters for the Spectrogram layer as per the example
        self.num_freqs = 257
        self.max_freq = 10_000

        # Conv1D layer after Spectrogram to mimic the example model
        self.conv1d = tf.keras.layers.Conv1D(16, 3)
    
    def call(self, inputs):
        # inputs: tuple or list of two tensors: (signal, fs)
        x, fs = inputs

        # Compute nfft based on sampling freq
        # Using first element in batch fs[0,0] as in example (assumes batch uniform fs)
        nfft = tf.cast(fs[0, 0] * (self.num_freqs - 1) / self.max_freq, tf.int32)

        # Compute STFT with nfft window size
        y = tf.signal.stft(x, frame_length=nfft, frame_step=256, fft_length=nfft, pad_end=True)
        # Take sqrt(abs()) and slice frequencies to num_freqs
        y = tf.sqrt(tf.abs(y))[:, :, :self.num_freqs]

        # IMPORTANT: set static shape explicitly so subsequent Conv1D layer has known channels dimension
        # The shape after STFT is (batch, time_frames, num_freqs)
        y.set_shape([None, None, self.num_freqs])

        # Pass through Conv1D layer (expects known channels)
        out = self.conv1d(y)
        return out

def my_model_function():
    # Return an instance of MyModel, no additional initialization
    return MyModel()

def GetInput():
    # Generate a random input tuple (signal, fs) matching model input expectations:
    # signal: shape (B, T), batch of 1D signals with variable length T
    # fs: shape (B, 1), batch of sampling frequencies

    B = 2  # batch size
    T = 16000  # example arbitrary input length in samples

    # Random signals in range [-1,1] float32
    signal = tf.random.uniform((B, T), minval=-1.0, maxval=1.0, dtype=tf.float32)
    # Sampling frequencies (e.g. 8000 or 16000 Hz) as float32, shape (B,1)
    fs = tf.constant([[8000.0], [16000.0]], dtype=tf.float32)
    return (signal, fs)

