# tf.random.uniform((640, 4096), dtype=tf.float32) ← inferred input shape from example code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward model similar to the example
        self.dense1 = tf.keras.layers.Dense(2048)
        self.dense2 = tf.keras.layers.Dense(256)
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output


def my_model_function():
    # Instantiate and return the model
    return MyModel()


def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # From the issue's test data: shape [640, 4096], dtype float32
    return tf.random.uniform((640, 4096), dtype=tf.float32)

