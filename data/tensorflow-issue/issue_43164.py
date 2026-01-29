# tf.random.uniform((B,), dtype=tf.float32) ← input is a batch of scalar floats

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential-like architecture from the issue:
        # Input(1) -> Dense(10) -> Activation('sigmoid') -> Dense(1)
        self.dense1 = tf.keras.layers.Dense(10)
        self.act = tf.keras.layers.Activation('sigmoid')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.act(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # The original example used 1D inputs with 10 samples of scalar values.
    # To keep flexible for batch size, we return a 1D tensor with shape (batch_size, 1).
    # Here, batch_size=10 to match the original example.
    # Inputs are scalar floats sampled from a normal distribution.
    batch_size = 10
    # Generate input tensor of shape (batch_size, 1), normal distribution ~ N(1,2)
    x = tf.random.normal(shape=(batch_size, 1), mean=1.0, stddev=2.0, dtype=tf.float32)
    return x

