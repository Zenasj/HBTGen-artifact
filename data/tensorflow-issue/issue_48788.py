# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from example: inputs are 1D samples with shape (batch_size, 1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    A fused model encapsulating two behaviors:
    1) A simple Keras model performing a single dense tanh layer followed by a dense output layer.
    2) A SciANN-like functional model mimicking a similar architecture for demonstration.

    The call returns a dictionary with outputs of both models and their MSE difference,
    serving as a comparison demonstration between two "model" styles in one model.

    Assumptions made:
    - Input: (batch_size, 1)
    - The SciANN model is emulated as a functional style model within the subclass.
    - The output is a dict containing outputs and their difference to illustrate comparison.
    """
    def __init__(self):
        super().__init__()
        # Standard Keras style model layers
        self.dense1 = tf.keras.layers.Dense(10, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(1)
        # Emulated "SciANN" style functional layers
        # In absence of SciANN package, we mimic with separate layers
        self.sc_dense1 = tf.keras.layers.Dense(10, activation='tanh')
        self.sc_dense2 = tf.keras.layers.Dense(1)
        # MSE loss component to compare outputs
        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, inputs, training=False):
        # Keras model forward
        keras_out1 = self.dense1(inputs)
        keras_out2 = self.dense2(keras_out1)

        # SciANN emulated forward - separate layers but same input
        sciann_out1 = self.sc_dense1(inputs)
        sciann_out2 = self.sc_dense2(sciann_out1)

        # Compute MSE difference (elementwise per sample)
        mse_diff = self.mse(keras_out2, sciann_out2)

        # Return dictionary with outputs and difference for potential comparison or debugging
        return {
            'keras_output': keras_out2,
            'sciann_output': sciann_out2,
            'mse_difference': mse_diff
        }

def my_model_function():
    # Return an instance of MyModel with initialized weights
    model = MyModel()
    # Build model once with dummy input to initialize weights and track shapes
    dummy_input = tf.zeros((1,1), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Generate random uniform input tensor matching expected shape (batch_size, features=1)
    # batch size reasonable for demonstration/testing, e.g. 32
    batch_size = 32
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

