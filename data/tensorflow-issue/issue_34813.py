# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape from the original Keras Input shape=(28, 28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define keras layers exactly as in the issue's model
        # Input shape: (28, 28)
        # GRU with 154 units
        self.gru = tf.keras.layers.GRU(154)
        self.flatten = tf.keras.layers.Flatten()
        # Dense with 10 units, relu activation
        self.dense = tf.keras.layers.Dense(10, activation="relu")

    def call(self, inputs):
        x = self.gru(inputs)
        x = self.flatten(x)
        out = self.dense(x)
        return out


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching the input shape (batch, 28, 28)
    # Batch size here chosen arbitrarily as 1 for simplicity.
    # dtype float32 (default for tf.random.uniform)
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

