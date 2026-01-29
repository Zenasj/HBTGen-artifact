# tf.random.uniform((B, 32), dtype=tf.float32) ← Input shape inferred from example: (batch_size, 32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original example model from the issue, a 3-layer dense model
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    """
    Return an instance of MyModel.
    No pre-trained weights are provided since the issue example trains from scratch.
    """
    return MyModel()

def GetInput():
    """
    Generates a random input tensor matching expected input of shape (batch_size, 32).
    Use a batch size of 8 as a reasonable default.
    """
    batch_size = 8
    input_shape = (batch_size, 32)
    x = tf.random.uniform(input_shape, dtype=tf.float32)
    return x

