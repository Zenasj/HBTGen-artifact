# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Input shape inferred from MNIST dataset loading in original example

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # KL divergence function scaled by dataset size (used for Bayesian layers)
        self.num_train = 60000  # MNIST train dataset size, inferred from code
        self.kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / tf.cast(self.num_train, tf.float32)

        # Flatten the (28, 28) input image to vector
        self.flatten = tf.keras.layers.Flatten()

        # Bayesian Dense layer with Flipout estimator (from tfp)
        self.dense_flipout = tfp.layers.DenseFlipout(
            10,
            kernel_divergence_fn=self.kl_divergence_function,
            activation=tf.nn.softmax
        )

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        # Apply DenseFlipout with softmax activation
        x = self.dense_flipout(x, training=training)
        return x


def my_model_function():
    # Return an instance of MyModel, no special initialization needed
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the model expected input shape.
    # Original input shape: MNIST images 28x28 (grayscale, no channel dimension)
    # Use batch size 1 as example
    batch_size = 1
    height = 28
    width = 28
    dtype = tf.float32
    # Values in range [0,1], float32 matching original preprocessed input
    return tf.random.uniform((batch_size, height, width), 0.0, 1.0, dtype=dtype)

