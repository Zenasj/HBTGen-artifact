import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class MFCCLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MFCCLayer, self).__init__(**kwargs)

    def call(self, pcm):
        # A 1024-point STFT with frames of 64 ms and 75% overlap.
        stfts = tf.signal.stft(pcm, frame_length=1024, frame_step=256, fft_length=1024)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first 13.
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[
            ..., :13
        ]
        print("mfccs.shape: ", mfccs.shape)
        return mfccs


def build_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    output_layer = MFCCLayer()(input_layer)
    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


if __name__ == "__main__":
    batch_size, num_samples, sample_rate = 32, 32000, 16000.0
    # A Tensor of [batch_size, num_samples] mono PCM samples in the range [-1, 1].
    pcm = tf.random.normal([batch_size, num_samples], dtype=tf.float32)
    print("pcm.shape: ", pcm.shape)

    model = build_model(pcm.shape)
    model.summary()

    # Convert to TensorFlow Lite and Save
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open("mfcc.tflite", "wb") as f:
        f.write(tflite_model)

    # Load the model and run inference
    with open("mfcc.tflite", "rb") as f:
        tflite_model = f.read()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    pcm = tf.expand_dims(pcm, axis=0)  # Add batch dimension

    interpreter.set_tensor(input_details[0]["index"], pcm)
    interpreter.invoke()  # <-- RuntimeError: tensorflow/lite/kernels/rfft2d.cc:117 IsPowerOfTwo(fft_length_data[1]) was not true.Node number 42 (RFFT2D) failed to prepare.
    mfccs = interpreter.get_tensor(output_details[0]["index"])
    print("mfccs.shape: ", mfccs.shape)