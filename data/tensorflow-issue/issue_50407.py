# tf.random.uniform((B, 1), dtype=tf.float32) ← The model input shape is (batch_size, 1)

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

# Custom negative log likelihood loss function for the probabilistic model
def negloglik_loss(y_true, y_pred):
    # y_pred is a tuple (nll, dist, x_mu)
    # We only use nll for loss calculation
    nll, _, _ = y_pred
    return nll

# Custom negative log likelihood metric function
def negloglik_metric(y_true, y_pred):
    # Same extraction of nll from y_pred tuple
    nll, _, _ = y_pred
    return nll


class MyModel(tf.keras.Model):
    """
    Probabilistic regression model using TensorFlow Probability distributions.
    The model returns a tuple of (negative log likelihood, distribution object, mu prediction).
    
    Inputs:
      Tuple of (input_x, input_y), both (batch_size, 1)
    Outputs:
      Tuple of (nll, distribution, mu), where:
        nll: Negative log likelihood tensor of shape (batch_size,)
        distribution: tfp.distributions.Normal instance with parameters from the model
        mu: predicted mean tensor of shape (batch_size, 1)
    
    Notes:
    - This model is adapted from a Google Colab TensorFlow Probability tutorial.
    - The negative log likelihood is used as loss and metric.
    - This design facilitates returning the learned distribution directly.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # One dense layer with relu activation to produce mean prediction mu
        self.block_1 = tf.keras.layers.Dense(1, activation='relu')

        # DistributionLambda layer to map mu to Normal distribution with fixed scale=1
        self.dist_lambda = tfp.layers.DistributionLambda(
            make_distribution_fn=lambda x: tfd.Normal(loc=x, scale=1)
        )

    def call(self, inputs):
        # Expect inputs: tuple/list (input_x, input_y)
        input_x, input_y = inputs  # Both shape: (batch_size, 1)

        x_mu = self.block_1(input_x)  # predicted mean (batch_size, 1)
        dist = self.dist_lambda(x_mu)  # distribution over outputs

        nll = -dist.log_prob(input_y)  # negative log likelihood of true labels

        # Return tuple (nll, distribution, mu) for loss and metrics
        return nll, dist, x_mu


def my_model_function():
    """
    Creates the MyModel instance wrapped in a compiled Keras Model
    supporting custom loss and metric.

    Since the original code uses model subclass inside a Keras Functional Model
    (mixing input layers and subclassed model for distribution output),
    here we return the subclassed model directly.

    Users can create input tensors of shape (batch_size, 1) and call MyModel with inputs (x, y).
    Loss and metric must be handled externally or via custom training loop.
    """
    return MyModel()


def GetInput():
    """
    Returns a sample input tuple (input_x, input_y) for MyModel call,
    where both inputs have shape (batch_size, 1).

    Using batch size of 16 for example, with float32 data.
    """
    B = 16
    x = tf.random.uniform((B, 1), minval=-20.0, maxval=60.0, dtype=tf.float32)
    # For y (labels), we synthesize target values using the original formula with noise
    # as defined in the original dataset loading function.

    # Parameters from original data generation
    w0 = 0.125
    b0 = 5.0
    x_range = [-20, 60]

    def s(x):
        # scale function from x
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + tf.square(g))

    # Generate true mean
    mu = w0 * x * (1.0 + tf.sin(x)) + b0

    # Add Gaussian noise scaled by s(x)
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=1.0) * s(x)

    y = mu + noise  # noisy targets shape (B, 1)

    return (x, y)

