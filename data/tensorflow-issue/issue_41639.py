# tf.random.uniform((B, 16), dtype=tf.float32) ← Based on input_shape=(16,), batch size varies in example (e.g., 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two Dense layers as per the example: first Dense(8), then Dense(1)
        self.dense1 = tf.keras.layers.Dense(8)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    """
    Returns an instance of MyModel.
    Weight initialization is default.
    """
    return MyModel()

def GetInput():
    """
    Returns a random input tensor compatible with MyModel.
    Shape: (batch_size=32, input_dim=16)
    """
    return tf.random.uniform((32, 16), dtype=tf.float32)

