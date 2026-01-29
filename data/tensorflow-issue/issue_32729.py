# tf.random.normal((B, D)) ← The input shape is assumed to be (batch_size, input_dim), where D = input feature dimension

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, units=3, std_init=0.5):
        """
        A custom NoisyDense layer integrated as a Keras Model.
        This replicates the behavior of the NoisyDense layer from the issue.

        units: output dimension
        std_init: initial stddev for noise scale parameters
        """
        super().__init__()
        self.units = units
        self.std_init = std_init
        # Variables for kernel and bias components for mu and sigma, and noise variables
        # Will be initialized in build using input shape

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.reset_noise(input_dim)
        mu_range = 1 / np.sqrt(input_dim)
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        # Define trainable variables for weight_mu, weight_sigma, bias_mu, bias_sigma
        self.weight_mu = tf.Variable(
            initial_value=mu_initializer(shape=(input_dim, self.units), dtype='float32'),
            trainable=True, name='weight_mu'
        )
        self.weight_sigma = tf.Variable(
            initial_value=sigma_initializer(shape=(input_dim, self.units), dtype='float32'),
            trainable=True, name='weight_sigma'
        )
        self.bias_mu = tf.Variable(
            initial_value=mu_initializer(shape=(self.units,), dtype='float32'),
            trainable=True, name='bias_mu'
        )
        self.bias_sigma = tf.Variable(
            initial_value=sigma_initializer(shape=(self.units,), dtype='float32'),
            trainable=True, name='bias_sigma'
        )

        # Note: Do NOT compute self.kernel or self.bias as variables here because
        # the noise depends on random variables that should be refreshed each call.
        # Instead, combine them during the call() method for correct gradients.
        self.built = True

    def call(self, inputs):
        # Compute noisy weights and biases on every call, so gradient can flow correctly,
        # as the noise variables depend on random values
        self.reset_noise(tf.shape(inputs)[-1])  # Reset noise for current input dimension
        
        kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        bias = self.bias_mu + self.bias_sigma * self.bias_eps
        
        return tf.matmul(inputs, kernel) + bias

    def _scale_noise(self, dim):
        # Generates factorized Gaussian noise for noise variables
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise) + 1e-12)  # Added epsilon to avoid sqrt(0)

    def reset_noise(self, input_dim):
        # Create noise tensors for weights and biases based on input and output dimensions
        eps_in = self._scale_noise(input_dim)
        eps_out = self._scale_noise(self.units)
        # Factorized noise for weights: outer product of eps_in and eps_out
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out

def my_model_function():
    # Create and return an instance of MyModel with default parameters
    return MyModel()

def GetInput():
    # Generate a batch of random inputs compatible with MyModel
    # Assume batch size 4 and input dimension 5 (arbitrary chosen)
    B, D = 4, 5
    return tf.random.uniform((B, D), dtype=tf.float32)

