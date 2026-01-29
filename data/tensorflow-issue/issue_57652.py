# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape is (batch_size, 1) since input shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward model mimicking the described architecture:
        # Input layer shape (1,), Dense 4 units with sigmoid, Dense 1 output unit, no activation on output.
        self.dense1 = tf.keras.layers.Dense(
            4,
            activation="sigmoid",
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=1),
        )
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (batch_size, 1), compatible with MyModel
    # Using batch size 32 as an example
    batch_size = 32
    # Uniform float values in range [-1,1] to mimic training input domain
    return tf.random.uniform((batch_size, 1), minval=-1.0, maxval=1.0, dtype=tf.float32)

