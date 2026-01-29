# tf.random.uniform((B, S_dim), dtype=tf.float32) ← inferred input shape: batch dimension unknown, obs_dim = S_dim

import tensorflow as tf
import tensorflow_probability as tfp

# Constants used in squashing and log sigma clipping (inferred typical values, missing in original)
LOG_SIGMA_MIN_MAX = (-20.0, 2.0)

# SquashBijector implements a tanh transformation to squash action outputs between [-1,1]
class SquashBijector(tfp.bijectors.Bijector):
    def __init__(self, validate_args=False, name="squash"):
        super().__init__(
            forward_min_event_ndims=0,
            validate_args=validate_args,
            name=name,
            inverse_min_event_ndims=0,
        )

    def _forward(self, x):
        return tf.tanh(x)

    def _inverse(self, y):
        # Clip y to prevent numerical issues with atanh near boundaries
        y = tf.clip_by_value(y, -0.99999997, 0.99999997)
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        # log|det(d tanh / dx)| = sum over dims of log(1 - tanh(x)^2)
        return tf.reduce_sum(
            tf.math.log1p(-tf.tanh(x) ** 2 + 1e-6), axis=-1
        )  # small epsilon for numerical stability

# The combined model class encapsulates the original SquashedGaussianActor.
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # For demonstration, use inferred dimensions and hidden sizes from original code snippet
        # Since obs_dim and act_dim are parameters in original, we use placeholders here.
        # We assume:
        # obs_dim = 6 (inferred from grad arrays in issue)
        # act_dim = 2 (inferred from grad arrays in issue)
        # hidden_sizes = [6, 6] as per implicit from network layers info
        self.s_dim = 6
        self.a_dim = 2
        self.hidden_sizes = [6, 6]

        # Seeds for reproducibility (inferred, using None)
        self._seed = None
        self._tfp_seed = None

        self._initializer = tf.keras.initializers.GlorotUniform(seed=self._seed)

        # Build fully connected layers sequentially
        self.net = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=(self.s_dim,))]
        )
        for i, hidden_size_i in enumerate(self.hidden_sizes):
            self.net.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    kernel_initializer=self._initializer,
                    name=f"l{i+1}",
                )
            )

        # Mu output head: dense layer without activation
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=(self.hidden_sizes[-1],)),
                tf.keras.layers.Dense(
                    self.a_dim,
                    activation=None,
                    kernel_initializer=self._initializer,
                    name="mu",
                ),
            ]
        )

        # Log sigma output head: dense layer without activation
        self.log_sigma = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(dtype=tf.float32, input_shape=(self.hidden_sizes[-1],)),
                tf.keras.layers.Dense(
                    self.a_dim,
                    activation=None,
                    kernel_initializer=self._initializer,
                    name="log_sigma",
                ),
            ]
        )

        # Squash bijector instance
        self.squash_bijector = SquashBijector()

    @tf.function
    def call(self, inputs):
        """
        Forward pass of the Squashed Gaussian Actor network with reparameterization trick.

        Inputs:
            inputs: Tensor of shape [batch_size, s_dim]
        Returns:
            clipped_action: squashed sampled action 
            clipped_mu: squashed mean action (deterministic)
            log_prob: log probability of clipped_action under policy
            epsilon: noise sample used for reparameterization (for debugging or further use)
        """
        obs = inputs

        # Forward pass through fully connected layers
        net_out = self.net(obs)

        # Compute mu and log_sigma
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = tf.clip_by_value(log_sigma, LOG_SIGMA_MIN_MAX[0], LOG_SIGMA_MIN_MAX[1])
        sigma = tf.exp(log_sigma)

        # Create bijectors for reparameterization
        affine_bijector = tfp.bijectors.Chain([
            tfp.bijectors.Shift(mu),
            tfp.bijectors.Scale(sigma)
        ])

        # Construct base normal distribution for sampling
        batch_size = tf.shape(obs)[0]
        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
        )

        # Sample epsilon noise for reparameterization
        epsilon = base_distribution.sample(batch_size, seed=self._tfp_seed)

        # Apply affine transform and squash
        raw_action = affine_bijector.forward(epsilon)
        clipped_action = self.squash_bijector.forward(raw_action)

        # Construct transformed distribution object for log_prob computation
        transform_bijector = tfp.bijectors.Chain([self.squash_bijector, affine_bijector])
        transformed_distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution,
            bijector=transform_bijector,
        )

        # Squashed deterministic policy mean
        clipped_mu = self.squash_bijector.forward(mu)

        # Compute log probability of sampled action
        log_prob = transformed_distribution.log_prob(clipped_action)

        return clipped_action, clipped_mu, log_prob, epsilon


def my_model_function():
    # Return an instance of MyModel. No special initialization parameters provided.
    return MyModel()


def GetInput():
    # Return a random tensor input compatible with MyModel input: shape = (batch_size, s_dim)
    # Assuming batch_size = 4 for example, s_dim = 6 (inferred)
    batch_size = 4
    s_dim = 6
    input_tensor = tf.random.uniform((batch_size, s_dim), dtype=tf.float32)
    return input_tensor

