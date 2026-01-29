# tf.random.uniform((B, 2), dtype=tf.float64) ← Input shape inferred from tests (batch size arbitrary, width 2), dtype float64 as in examples

import tensorflow as tf
from tensorflow import keras
K = keras.backend

class MyLayer(keras.layers.Layer):
    def call(self, inputs, training=None):
        # Note: Avoid using @tf.function here because it causes learning_phase to be cached,
        # breaking dynamic behavior with training argument.
        # This matches the issue and resolution discussion.
        tf.print("training: ", training)
        tf.print("K.learning_phase(): ", K.learning_phase())
        # Use keras.backend.in_test_phase to simulate different behavior in train/test mode
        return keras.backend.in_test_phase(inputs + 1., inputs + 2., training)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model is a Sequential wrapper around MyLayer with input shape [2]
        self.seq = keras.models.Sequential([MyLayer(input_shape=[2])])

    def call(self, inputs, training=None):
        # Forward inputs to the Sequential model, passing training flag explicitly.
        # This allows the MyLayer to see the training argument correctly.
        return self.seq(inputs, training=training)

def my_model_function():
    # Returns an instance of MyModel as defined.
    return MyModel()

def GetInput():
    # Return a valid input tensor matching expected input shape and dtype.
    # From the issue inputs seem to be float64, shape (batch_size, 2).
    # Using batch_size = 1 for simplicity.
    return tf.random.uniform((1, 2), dtype=tf.float64)

