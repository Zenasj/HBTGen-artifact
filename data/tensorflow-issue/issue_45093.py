# tf.random.uniform((samples,), dtype=tf.float32) ← Input is a scalar int64 tensor indicating number of samples

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tf.math

class MyModel(tf.keras.Model):
    def __init__(self, max_samples=100):
        super().__init__()
        # Store max_samples as max capacity for state Variables
        self.max_samples = max_samples

        # Variables to hold state across tf.function calls
        # Use int64 for GPU compatibility as noted in the issue discussion
        self.i = tf.Variable(0, dtype=tf.int64, trainable=False)
        # Pre-allocate tensor to hold log-likelihood values, zeros initial
        self.log_likes = tf.Variable(tf.zeros([max_samples], dtype=tf.float32), trainable=False)
        # Scalar variable to hold mean log_likelihood across runs
        self.log_like = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    @tf.function(jit_compile=True)
    def call(self, samples):
        # samples: scalar tf.Tensor int64 number of samples to process
        
        # Inner sampling function: sum of two independent Normals
        def rnd():
            return tfd.Normal(0., 1.).sample() + tfd.Normal(3., 1.).sample()

        # Initialize a TensorArray to hold per-sample log-likelihoods
        # Size initially samples converted to int32 (required by TensorArray API)
        x = tf.TensorArray(tf.float32, size=tf.cast(samples, tf.int32))
        # Read current iteration index from self.i (state Variable)
        i = self.i.read_value()

        # Unstack previously stored values of log_likes into x
        # Note: x must be large enough to hold all samples to avoid shape errors
        x = x.unstack(self.log_likes.read_value())

        # Loop from i until samples, computing new log-likelihood samples
        while i < samples:
            # Write the computed log-prob at index i and assign back to x
            x = x.write(
                tf.cast(i, tf.int32),
                tfm.reduce_sum(tfd.Normal(rnd(), 1.).log_prob(0.4))
            )
            i += 1

        # Stack the TensorArray contents to get the current log-likes vector
        log_likes = x.stack()

        # Compute mean log likelihood across all samples
        log_like = tfm.reduce_mean(log_likes)

        # loss defined as mean log_like (same here, scalar)
        loss = tfm.reduce_mean(log_like)

        # Update variables with current iteration, padded log_likes and mean log_like
        self.i.assign(i)

        # Pad log_likes if samples < max_samples so assignment shape matches variable shape
        padded = tf.pad(log_likes, [[0, self.max_samples - samples]])
        self.log_likes.assign(padded)

        self.log_like.assign(log_like)

        return loss


def my_model_function():
    # Return an instance of MyModel with default max_samples=100
    return MyModel()


def GetInput():
    # Return input tensor compatible with MyModel's __call__
    # It expects a scalar int64 tensor indicating number of samples to process
    # For demonstration, return 5 samples
    return tf.constant(5, dtype=tf.int64)

