# tf.random.uniform((B, 40000), dtype=tf.float32)
import tensorflow as tf

def extract_features(inputs):
    sample_rate = 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    # inputs shape: (batch_size, 40000)
    stfts = tf.signal.stft(inputs, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)  # magnitude spectrogram

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms, returning first 40 coefficients (common choice for good frequency coverage)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :40]
    
    return mfccs

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use a Lambda layer to encapsulate the extraction function as a layer
        self.mfcc_layer = tf.keras.layers.Lambda(extract_features)
    
    def call(self, x):
        return self.mfcc_layer(x)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Build the model explicitly for shape (None, 40000)
    model.build(input_shape=(None, 40000))

    # Normally weights not needed here since Lambda layer has no trainable weights
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # According to code, input is (batch_size, 40000) float32 tensor resembling audio waveform
    # Use batch size = a small number like 2 for testing
    batch_size = 2
    input_tensor = tf.random.uniform((batch_size, 40000), dtype=tf.float32)
    return input_tensor

