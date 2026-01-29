# tf.random.uniform((B, input_dim), dtype=tf.float32) ← assuming input is a batch of feature vectors of some size

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_models=3, output_dim=1):
        """
        An ensemble of MLP models producing two outputs per model: mean and variance.
        Args:
          num_models: Number of sub-models in the ensemble.
          output_dim: Dimensionality of the output (e.g., regression scalar).
        """
        super(MyModel, self).__init__()
        self.num_models = num_models
        
        # Each model's mean predictor: a small feed-forward network
        self.mean = [tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim)
        ]) for _ in range(num_models)]
        
        # Each model's variance predictor: must produce positive variance - use softplus activation
        self.variance = [tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='softplus')  # softplus ensures positivity
        ]) for _ in range(num_models)]

    def call(self, inputs, training=None, mask=None):
        mean_predictions = []
        variance_predictions = []
        # Looping over ensemble submodels 
        for idx in range(self.num_models):
            mean_pred = self.mean[idx](inputs, training=training)
            var_pred = self.variance[idx](inputs, training=training)
            mean_predictions.append(mean_pred)
            variance_predictions.append(var_pred)
        
        # Stack results: shape (num_models, batch_size, output_dim)
        mean_stack = tf.stack(mean_predictions, axis=0)
        variance_stack = tf.stack(variance_predictions, axis=0)
        
        return mean_stack, variance_stack


class GaussianNLL(tf.keras.losses.Loss):
    def __init__(self):
        super(GaussianNLL, self).__init__()
        
    def call(self, y_true, y_pred):
        """
        y_pred is a tuple (mean, variance) where each has shape (num_models, batch, output_dim)
        y_true has shape (batch, output_dim)
        This loss computes the negative log-likelihood of y_true under a Gaussian with predicted mean and variance,
        then averages over ensemble models and batch.
        """
        mean, variance = y_pred
        
        # Add small epsilon for numerical stability
        variance = variance + 1e-4
        
        # Expand y_true to match mean shape: (1, batch, output_dim) broadcast to (num_models, batch, output_dim)
        y_true_expanded = tf.expand_dims(y_true, axis=0)
        
        # Gaussian negative log likelihood per element
        nll_elementwise = 0.5 * tf.math.log(variance) + 0.5 * ((y_true_expanded - mean) ** 2) / variance
        
        # Reduce mean over output_dim and batch, then average over ensemble models
        nll_per_model = tf.reduce_mean(nll_elementwise, axis=[1,2])  # mean over batch and output_dim
        nll = tf.reduce_mean(nll_per_model)  # mean over models
        
        return nll


def my_model_function():
    """
    Returns an instance of MyModel.
    Default to 3 ensemble models and 1-d output.
    """
    model = MyModel(num_models=3, output_dim=1)
    return model


def GetInput():
    """
    Returns a random input tensor compatible with MyModel:
    Assuming input_dim=10 (can be adjusted to desired feature size)
    Produces a batch of 32 examples.
    """
    batch_size = 32
    input_dim = 10
    # Shape: (batch_size, input_dim)
    return tf.random.uniform((batch_size, input_dim), dtype=tf.float32)

