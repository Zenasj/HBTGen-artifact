import numpy as np
import random
import tensorflow as tf

self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
self.bias = self.bias_mu + self.bias_sigma * self.bias_eps

class NoisyDense(kl.Layer):
    def __init__(self, units, std_init=0.5):
        super().__init__()
        self.units = units
        self.std_init = std_init

    def build(self, input_shape):
        self.reset_noise(input_shape[-1])
        mu_range = 1 / np.sqrt(input_shape[-1])
        mu_initializer = tf.random_uniform_initializer(-mu_range, mu_range)
        sigma_initializer = tf.constant_initializer(self.std_init / np.sqrt(self.units))

        self.weight_mu = tf.Variable(initial_value=mu_initializer(shape=(input_shape[-1], self.units), dtype='float32'),
                                     trainable=True)

        self.weight_sigma = tf.Variable(initial_value=sigma_initializer(shape=(input_shape[-1], self.units), dtype='float32'),
                                        trainable=True)

        self.bias_mu = tf.Variable(initial_value=mu_initializer(shape=(self.units,), dtype='float32'),
                                     trainable=True)

        self.bias_sigma = tf.Variable(initial_value=sigma_initializer(shape=(self.units,), dtype='float32'),
                                        trainable=True)

        self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        self.bias = self.bias_mu + self.bias_sigma * self.bias_eps

    def call(self, inputs):
        # output = tf.tensordot(inputs, self.kernel, 1)
        # tf.nn.bias_add(output, self.bias)
        # return output
        # self.kernel = self.weight_mu + self.weight_sigma * self.weights_eps
        # self.bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return tf.matmul(inputs, self.kernel) + self.bias

    def _scale_noise(self, dim):
        noise = tf.random.normal([dim])
        return tf.sign(noise) * tf.sqrt(tf.abs(noise))

    def reset_noise(self, input_shape):
        eps_in = self._scale_noise(input_shape)
        eps_out = self._scale_noise(self.units)
        self.weights_eps = tf.multiply(tf.expand_dims(eps_in, 1), eps_out)
        self.bias_eps = eps_out