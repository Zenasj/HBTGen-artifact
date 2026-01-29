# tf.random.uniform((16000, 1), dtype=tf.float32) ← inferred input shape from the decode_wav method

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # This model replicates the core logic of the LiteWaveHandle class that uses a TFLite Interpreter
        # Since direct tflite_runtime Interpreter use isn't compatible with tf.keras.Model,
        # here we mimic the input/output tensor handling and keyword detection logic
        #
        # Assumptions & notes:
        # - The original code loads a .tflite model for keyword spotting.
        # - The input shape is (16000, 1) float32 audio waveform.
        # - The output is a 1D softmax vector over labels.
        # - Due to tf.keras.Model constraints, we simulate the model with a simple placeholder network,
        #   as the original TFLite model architecture is unavailable.
        #
        # In a real-world scenario, you would convert the TFLite model to a tf.keras model or load a compatible model.
        #
        # For completeness, we provide a minimal dummy network that expects the same input shape.

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(12, activation='softmax')  # 12 labels as per original labels

    def call(self, inputs):
        x = self.flatten(inputs)  # flatten 16000x1 -> 16000
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def detect_keywords(self, inputs, top_k=1):
        """Run inference and return top_k (label id, probability) tuples."""
        probs = self(inputs)
        # Get top_k indices and probabilities
        top_indices = tf.math.top_k(probs, k=top_k).indices.numpy()
        top_values = tf.math.top_k(probs, k=top_k).values.numpy()
        return list(zip(top_indices, top_values))


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()

    # Normally, you would load pretrained weights here if available.
    # Since the original TFLite model isn't loadable directly, we keep the randomly initialized weights.

    return model


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the original wave decode size: 16000 samples with 1 channel
    # The dtype is float32 as inferred from decode_wav conversion

    # Generate uniform random floats in range [-1.0, 1.0] to mimic typical normalized audio waveform
    input_tensor = tf.random.uniform((16000, 1), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return input_tensor

