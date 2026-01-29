# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape corresponds to MNIST images as used in the example

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assumptions:
        # - Input shape per sample: (28, 28) grayscale image (MNIST)
        # - Using DenseFlipout from tfp for Bayesian dense layer with kl_divergence regularization
        # - Output 10 classes with softmax activation
        
        # Flatten layer to convert 28x28 image to flat vector of length 784
        self.flatten = tf.keras.layers.Flatten()
        
        # KL divergence function normalized by the training set size (60000 as in MNIST)
        self.kl_divergence_function = lambda q, p, _: tfd.kl_divergence(q, p) / 60000.0
        
        # Bayesian dense layer (flipout variational inference)
        self.dense_flipout = tfp.layers.DenseFlipout(
            10,
            kernel_divergence_fn=self.kl_divergence_function,
            activation='softmax'
        )
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        output = self.dense_flipout(x, training=training)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random batch of MNIST-like images for testing
    # Batch size chosen arbitrarily as 32
    # dtype float32 normalized [0, 1]
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

