# tf.random.normal((B, 112, 112, 3), dtype=tf.float32) ← Input shape inferred from data tensor in issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the issue's get_net Sequential model but reconstructing correctly
        self.conv = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3))
        # The Dense layer after Conv2D needs flattening or global pooling - assumed flattening here
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.flatten(x)
        output = self.dense(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching the input expected by MyModel:
    # Shape: [batch_size, 112, 112, 3] as per issue data example
    # Using batch size similar to example (e.g., 80 or 128)
    batch_size = 80
    return tf.random.normal(shape=(batch_size, 112, 112, 3), dtype=tf.float32)

