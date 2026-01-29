# tf.random.uniform((B, 1000), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same Dense layers as in the issue for reproducing the example model
        self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(10000, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(10000, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(1000, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


def my_model_function():
    # Instantiate MyModel with default initialization, no pretrained weights
    return MyModel()


def GetInput():
    # Based on the input shape (B, 1000) expected by the first Dense layer,
    # create a random input with batch size 64 and feature size 1000.
    # The original code used variable batch sizes for datasets (like 64 or 64*20 total samples),
    # but batch is the dynamic dimension anyway.
    # Use float32 as dtype for typical tf.keras model input.
    return tf.random.uniform((64, 1000), dtype=tf.float32)

