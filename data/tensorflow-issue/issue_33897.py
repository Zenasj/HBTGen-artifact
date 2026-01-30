import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def generate_mfcc_features(audio_tensor):
    sample_rate = 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(audio_tensor, frame_length=1024, frame_step=256,fft_length=1024)
    spectrograms = tf.cast(tf.abs(stfts), tf.float32)

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

converter = tf.lite.TFLiteConverter.from_concrete_functions([enc_to_save])
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = set([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
tflite_model_enc = converter.convert()

def extract_features(inputs):
    sample_rate = 16000.0

    # A 1024-point STFT with frames of 64 ms and 75% overlap.
    stfts = tf.signal.stft(inputs, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :40]
    
    return log_mel_spectrograms

class NN(tf.keras.Model):
    def __init__(self):
        super(NN, self).__init__()
        
        self.mfcc = tf.keras.layers.Lambda(extract_features)

    def call(self, x):    
        return self.mfcc(x)
    
nn = NN()
nn.build((None,40000))

@tf.function
def evaluate(inp):
    return nn(inp)

input_shape = tf.TensorSpec([None,40000], tf.float32)
to_save = evaluate.get_concrete_function(input_shape)

converter = tf.lite.TFLiteConverter.from_concrete_functions([to_save])
# converter.experimental_new_converter = True
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
converter.target_spec.supported_ops = set([tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], tf.convert_to_tensor(np.expand_dims(audio_data, 0), dtype=tf.float32))

interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])