# tf.random.uniform((B,), dtype=tf.float32) ← Input shape inferred as a 1D audio waveform vector, here B=40000

import tensorflow as tf

def generate_mfcc_features(audio_array):
    sample_rate = 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(audio_array, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)  # Magnitude spectrogram

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    # Cast to float32 explicitly for TFLite compatibility and stable ops.
    linear_to_mel_weight_matrix = tf.cast(linear_to_mel_weight_matrix, tf.float32)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, axes=1)
    # Set shape explicitly for static shape inference in TF graph / TFLite.
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13 coefficients.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]

    return mfccs

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Lambda layer to extract MFCC features from raw audio input
        self.mfcc_layer = tf.keras.layers.Lambda(generate_mfcc_features)
        # Conv1D layer analogous to given example, filters=32, kernel_size=10, strides=2, 'same' padding, relu activation
        self.conv1d = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=2, padding='same', activation='relu')

    def call(self, inputs):
        """
        inputs: Tensor of shape (batch_size, 40000) representing raw audio waveform(s)
        Returns: Tensor after MFCC extraction and Conv1D processing
        """
        # Extract MFCC features: shape => (batch_size, time_frames, 13)
        mfccs = self.mfcc_layer(inputs)
        # Apply Conv1D to the MFCC features
        outputs = self.conv1d(mfccs)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random batch of 1 audio example of length 40000, dtype float32
    # This matches the expected input shape and type of MyModel
    return tf.random.uniform(shape=(1, 40000), dtype=tf.float32)

