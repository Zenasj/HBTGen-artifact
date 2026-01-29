# tf.random.uniform((B, input_dim), dtype=tf.float32) ← assuming input shape (batch_size, input_dim)

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp
import numpy as np

# This global prior params must be set once for all layer instances
def mixture_prior_params(sigma_1, sigma_2, pi):
    # Create a tf.Variable for prior params to be trainable or trackable
    # Use float32 dtype consistent with TF2 default
    params = tf.Variable([sigma_1, sigma_2, pi], dtype=tf.float32, trainable=False, name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

# Initialize global prior parameters (shared across layers)
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

def log_mixture_prior_prob(w):
    # Mixture of two zero-mean Normals with stddev from prior_params
    comp_1_dist = tfp.distributions.Normal(loc=0.0, scale=prior_params[0])
    comp_2_dist = tfp.distributions.Normal(loc=0.0, scale=prior_params[1])
    comp_1_weight = prior_params[2]
    prob = comp_1_weight * comp_1_dist.prob(w) + (1.0 - comp_1_weight) * comp_2_dist.prob(w)
    # Add small epsilon to avoid log(0)
    return tf.math.log(prob + 1e-8)

class MyModel(tf.keras.Model):
    def __init__(self, output_dim=128, kl_loss_weight=1e-6, activation='relu'):
        super().__init__()
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation_fn = activations.get(activation)
        self.prior_params = prior_params  # reference to global prior

        # These weights will be created later in build(), but define placeholders here
        self.kernel_mu = None
        self.kernel_rho = None
        self.bias_mu = None
        self.bias_rho = None

    def build(self, input_shape):
        # input_shape: (batch_size, input_dim)
        input_dim = input_shape[-1]

        # We do NOT append prior_params as trainable weights here, prior_params is fixed variable

        # kernel_mu: mean of weight posterior distribution
        self.kernel_mu = self.add_weight(
            name='kernel_mu',
            shape=(input_dim, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=prior_sigma),
            trainable=True
        )
        # bias_mu: mean of bias posterior
        self.bias_mu = self.add_weight(
            name='bias_mu',
            shape=(self.output_dim,),
            initializer=tf.random_normal_initializer(stddev=prior_sigma),
            trainable=True
        )
        # kernel_rho: parameter to compute stddev for kernel posterior via softplus
        self.kernel_rho = self.add_weight(
            name='kernel_rho',
            shape=(input_dim, self.output_dim),
            initializer=tf.constant_initializer(-5.0),  # Initialized negative to have small initial stddev
            trainable=True
        )
        # bias_rho: parameter for bias stddev
        self.bias_rho = self.add_weight(
            name='bias_rho',
            shape=(self.output_dim,),
            initializer=tf.constant_initializer(-5.0),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Compute posterior stddevs via softplus to ensure positivity
        kernel_sigma = tf.math.softplus(self.kernel_rho)  # shape (input_dim, output_dim)
        bias_sigma = tf.math.softplus(self.bias_rho)      # shape (output_dim,)

        # Sample weights and biases from posterior Gaussian: reparameterization trick
        kernel_eps = tf.random.normal(shape=tf.shape(self.kernel_mu), dtype=self.kernel_mu.dtype)
        bias_eps = tf.random.normal(shape=tf.shape(self.bias_mu), dtype=self.bias_mu.dtype)
        sampled_kernel = self.kernel_mu + kernel_sigma * kernel_eps
        sampled_bias = self.bias_mu + bias_sigma * bias_eps

        # Add KL divergence loss weighted by kl_loss_weight to model loss
        self.add_loss(self.kl_loss(sampled_kernel, self.kernel_mu, kernel_sigma))
        self.add_loss(self.kl_loss(sampled_bias, self.bias_mu, bias_sigma))

        # Linear transformation with sampled weights and biases
        output = tf.linalg.matmul(inputs, sampled_kernel) + sampled_bias

        # Apply activation
        return self.activation_fn(output)

    def kl_loss(self, w, mu, sigma):
        # Compute KL divergence between variational posterior Normal(mu, sigma)
        # and mixture prior. KL = E_q[log q(w) - log p(w)]
        variational_dist = tfp.distributions.Normal(loc=mu, scale=sigma)
        log_qw = variational_dist.log_prob(w)
        log_pw = log_mixture_prior_prob(w)
        kl = tf.reduce_sum(log_qw - log_pw)
        return self.kl_loss_weight * kl

def my_model_function():
    # By default output_dim=128, kl_loss_weight=1e-6, activation='relu'
    return MyModel(output_dim=128, kl_loss_weight=1e-6, activation='relu')

def GetInput():
    # Generate a random input tensor of shape (batch_size, input_dim)
    # Assume input_dim = 64 for demonstration (not specified in issue)
    batch_size = 32
    input_dim = 64
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

