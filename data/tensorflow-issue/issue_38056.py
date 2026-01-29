# tf.random.uniform((B, 2), dtype=tf.float32), tf.random.uniform((B, 1), dtype=tf.float32), tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the Dense output layer to produce log-odds (linear activation)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='linear', name='output')

    def call(self, inputs):
        """
        inputs: tuple of three tensors:
          - predictors: shape (batch_size, 2), float32
          - N: shape (batch_size, 1), float32 (number of trials)
          - logNchooseK: shape (batch_size, 1), float32 (log of N choose k)
        Returns:
          log-odds predictions (shape (batch_size, 1), float32)
        """
        predictors, N, logNchooseK = inputs
        # Simple forward pass: compute linear output from predictors
        output = self.output_layer(predictors)
        return output

    def binomial_loss(self, y_true, y_pred, N, logNchooseK):
        """
        Custom binomial negative log likelihood loss function.

        y_true: true counts of successes (shape (batch_size, 1))
        y_pred: predicted log-odds (shape (batch_size, 1)) - output of model
        N: number of trials (shape (batch_size, 1))
        logNchooseK: log binomial coefficient (shape (batch_size, 1))
        """
        # convert log-odds to predicted probability
        predicted_prob = tf.math.exp(y_pred) / (1 + tf.math.exp(y_pred))
        # Avoid log(0) by clipping probabilities
        predicted_prob = tf.clip_by_value(predicted_prob, 1e-7, 1 - 1e-7)

        # Negative log likelihood of Binomial(k; N, p)
        neg_loglik = -(logNchooseK + y_true * tf.math.log(predicted_prob) + (N - y_true) * tf.math.log(1 - predicted_prob))
        return tf.reduce_mean(neg_loglik)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of random tensors that match the expected input shapes:
    # predictors: shape (batch_size, 2)
    # N: shape (batch_size,1)
    # logNchooseK: shape (batch_size,1)
    batch_size = 16
    predictors = tf.random.uniform((batch_size, 2), dtype=tf.float32)
    # N should represent number of trials, use integers in a reasonable range and cast to float32
    N = tf.random.uniform((batch_size, 1), minval=1, maxval=100, dtype=tf.int32)
    N = tf.cast(N, tf.float32)
    # logNchooseK should be float32, for dummy input generate uniform float values (approximate plausible)
    logNchooseK = tf.random.uniform((batch_size,1), minval=0, maxval=50, dtype=tf.float32)
    return (predictors, N, logNchooseK)

