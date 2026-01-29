# tf.random.uniform((N, 2), dtype=tf.float32) ← inferred input shape: batch of 2D vectors (N x 2)

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

class MyModel(tf.keras.Model):
    """
    This model implements a Multivariate Normal distribution parameterized by:

     - loc: mean vector
     - chol_precision_tril: lower-triangular Cholesky factor of the precision matrix

    It uses a bijector chain composed of:
     - a Shift bijector for the mean offset
     - an inverse ScaleMatvecTriL bijector representing the precision's lower-triangular
    
    Forward call returns samples from this distribution given some input (ignored).
    We define the call signature to accept inputs only to fit the tf.function jit_compile usage scenario.

    Since the original issue centers on the missing bijector Shift and Scale attributes in older tfp versions,
    here we assume TensorFlow Probability 0.12+ where tfb.Shift and tfb.ScaleMatvecTriL exist.
    """

    def __init__(self):
        super().__init__()

        # Fixed parameters as per the original code example:
        dtype = tf.float32
        self.loc = tf.constant([1., -1.], dtype=dtype)  # mean vector of shape (2,)
        self.chol_precision_tril = tf.constant([[1., 0.],
                                                [2., 8.]], dtype=dtype)  # shape (2,2)

        # Define the distribution with bijector transformations
        # Construct precision matrix from Cholesky factor:
        # Precision = chol_precision_tril @ chol_precision_tril.T
        # The covariance would be inv(precision)
        self.precision = tf.matmul(self.chol_precision_tril, self.chol_precision_tril, transpose_b=True)

        # Define the MVN distribution using tfd.TransformedDistribution and bijector chain.
        # The bijector chain is Shift then inverse ScaleMatvecTriL with adjoint=True,
        # meaning the scaling is by the transpose of the lower-triangular matrix.
        base_dist = tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(self.loc), scale=tf.ones_like(self.loc)),
            reinterpreted_batch_ndims=1)

        # Construct the bijector chain
        # Shift(shift=loc) moves the distribution by loc
        # Invert(ScaleMatvecTriL) applies the inverse of the linear transformation defined by precision's Chol lower-triangular matrix with adjoint=True
        bijector = tfb.Chain([
            tfb.Shift(shift=self.loc),
            tfb.Invert(tfb.ScaleMatvecTriL(scale_tril=self.chol_precision_tril, adjoint=True)),
        ])

        self.mvn_dist = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector, name="MVNCholPrecisionTriL")

    @tf.function
    def call(self, inputs):
        """
        Return samples from the defined MVN distribution.
        The inputs argument is unused but included to match keras Model call signature and to satisfy the @tf.function API.
        
        Args:
            inputs: tf.Tensor of shape (N, 2), ignored
        
        Returns:
            samples: tf.Tensor of shape (N, 2) sampled from MVN distribution
        """
        n = tf.shape(inputs)[0]
        samples = self.mvn_dist.sample(n)
        return samples


def my_model_function():
    # Return an instance of MyModel with the original fixed parameters for loc and chol_precision_tril
    return MyModel()


def GetInput():
    # Return a batch of random 2D input vectors with shape (N, 2), dtype float32
    # This input matches the expected input shape that MyModel.call() accepts, even though input is unused.
    N = 5  # example batch size
    return tf.random.uniform((N, 2), dtype=tf.float32)

