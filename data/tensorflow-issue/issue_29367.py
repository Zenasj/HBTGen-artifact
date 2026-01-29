# tf.random.uniform((3,))  ← inferred input shape from the example usage of mean vector with shape (3,)

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers

class MyModel(tf.keras.Model):
    """
    Combined model reflecting the concerns and patterns discussed:
    - Holds variables for mean and covariance factors.
    - Constructs LinearOperatorDiag or LinearOperatorLowerTriangular lazily (e.g. in call or build)
      to maintain the gradient path properly.
    - Initializes a MultivariateNormalTriL distribution with the covariance formed as L L^T.
    - Computes a loss based on samples from the distribution, ensuring all computations
      happen inside the gradient tape.
      
    This reflects the key idea from the issue:
    *Postpone construction of linear operators and distribution objects until inside the call() or build() method.*
    This way gradient tape can track operations properly.
    """
    def __init__(self, mean_init: tf.Tensor, covariance_factor_init: tf.Tensor):
        super().__init__()
        # Store initializers; variables will be created in build to be consistent with TF2 best practice.
        self.mean_init = mean_init
        self.cov_factor_init = covariance_factor_init
        
        # Placeholder for variables to be created at build time
        self.mean = None
        self.cov_factor = None

        # Placeholders for linear operators and distribution, will be constructed in call/build
        self.linop_cov = None
        self.distribution = None

    def build(self, input_shape=None):
        # Create variables on first call or build time
        # Note: Creating variables here avoids computations in __init__

        # Mean vector variable
        self.mean = self.add_weight(
            name="mean",
            shape=self.mean_init.shape,
            initializer=tf.keras.initializers.Constant(self.mean_init),
            trainable=True,
            dtype=tf.float32,
        )

        # Covariance factor variable (e.g. lower-triangular L for covariance = L L^T)
        # We assume input cov_factor_init is the Cholesky factor L of shape (D, D)
        self.cov_factor = self.add_weight(
            name="cov_factor",
            shape=self.cov_factor_init.shape,
            initializer=tf.keras.initializers.Constant(self.cov_factor_init),
            trainable=True,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(self, inputs=None):
        """
        Build LinearOperator and Distribution lazily in the call,
        based on current variables, so computations are tracked.
        
        Then sample from the distribution and compute a loss function from sample.
        """
        # Construct a LinearOperatorLowerTriangular from cov_factor variable:
        # This preserves gradient flow through the cov_factor variable.
        self.linop_cov = tf.linalg.LinearOperatorLowerTriangular(self.cov_factor)

        # Construct Multivariate Normal Triangular distribution using mean and linear operator
        # Using LinearOperator for covariance allows efficient computations, as in TFP.
        self.distribution = tfd.MultivariateNormalTriL(
            loc=self.mean,
            scale=self.linop_cov
        )

        # Sample from the distribution (note: sampling is differentiable with reparameterization)
        sample = self.distribution.sample()

        # Compute a simple loss as example: negative log probability of the sample
        # This is an example; real loss could depend on downstream tasks
        neg_log_prob = -self.distribution.log_prob(sample)

        return neg_log_prob


def my_model_function():
    """
    Returns an instance of MyModel initialized with example values as per the issue:
    - Mean vector of shape (3,)
    - Covariance factor L (lower-triangular) of shape (3, 3), from a random banded matrix
    """
    # Example initialization consistent with snippets from the issue
    mean_init = tf.random.uniform((3,), dtype=tf.float32)  # shape (3,)
    L_raw = tf.random.uniform((3, 3), dtype=tf.float32)
    L = tf.linalg.band_part(L_raw, -1, 0)  # lower triangular part of L_raw
    
    return MyModel(mean_init=mean_init, covariance_factor_init=L)


def GetInput():
    """
    Returns a valid input for MyModel.call().
    The MyModel.call() takes an optional input, but does not require it.
    Just return None or an empty tensor consistent with model signature.
    """

    # MyModel doesn't need input argument per current design, so return None
    return None

