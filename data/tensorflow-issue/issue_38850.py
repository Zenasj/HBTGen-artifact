# tf.random.uniform((B, 5), dtype=tf.float16) ← inferred input shape and dtype from issue example

import tensorflow as tf
from tensorflow.keras.layers import Dense, GaussianNoise, Activation, Input
from tensorflow.keras.models import Model


class MyModel(tf.keras.Model):
    def __init__(self, noise_stddev=0.01):
        super().__init__()
        # To illustrate core issue: GaussianNoise layer mixing float32 noise with float16 input
        # We explicitly set the GaussianNoise dtype to match the input dtype (float16) to avoid type errors
        self.dense1 = Dense(5, input_shape=(5,), dtype=tf.float16)
        # GaussianNoise usually uses float32 by default, 
        # but with mixed precision, input is float16 and noise must match dtype to avoid errors
        self.gaussian_noise = GaussianNoise(noise_stddev, dtype=tf.float16)
        self.activation = Activation("relu")
        self.dense2 = Dense(5, dtype=tf.float16)

    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        # GaussianNoise should work with matching dtype and respect training mode
        x = self.gaussian_noise(x, training=training)
        x = self.activation(x)
        x = self.dense2(x)
        return x


def my_model_function():
    # Returns an instance of MyModel, noise_stddev can be adjusted as needed
    return MyModel(noise_stddev=0.01)


def GetInput():
    # Return a random tensor input that matches MyModel's expected input:
    # shape=(batch_size, 5), dtype float16 for mixed precision compatibility
    # Batch size is arbitrarily chosen (e.g., 20 as in the example)
    return tf.random.uniform((20, 5), dtype=tf.float16)

