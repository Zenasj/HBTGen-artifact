# tf.random.uniform((B, 4, 1), dtype=tf.float32) ← Input shape inferred from data_x, data_y with shape (batch, 4, 1)

import tensorflow as tf

class WeightedSDRLoss(tf.keras.losses.Loss):
    def __init__(self, noisy_signal, reduction=tf.keras.losses.Reduction.AUTO, name='WeightedSDRLoss'):
        super().__init__(reduction=reduction, name=name)
        self.noisy_signal = noisy_signal

    def sdr_loss(self, sig_true, sig_pred):
        # Scale-invariant SDR-like loss: negative cosine similarity scaled by vector norms
        numerator = tf.reduce_mean(sig_true * sig_pred)
        denominator = tf.norm(sig_pred) * tf.norm(sig_true)
        return - numerator / denominator

    def call(self, y_true, y_pred):
        # noise_true and noise_pred represent noise components based on noisy_signal
        noise_true = self.noisy_signal - y_true
        noise_pred = self.noisy_signal - y_pred

        # Alpha balances contribution according to signal and noise power
        numerator = tf.reduce_mean(tf.square(y_true))
        denominator = tf.reduce_mean(tf.square(y_true)) + tf.reduce_mean(tf.square(self.noisy_signal - y_pred))
        alpha = numerator / denominator

        return alpha * self.sdr_loss(y_true, y_pred) + (1 - alpha) * self.sdr_loss(noise_true, noise_pred)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model accepts two inputs: 
        # x (input signal), y_true (ground truth for loss calculation)
        self.activation = tf.keras.layers.Activation('tanh')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=None):
        # Expect inputs as a tuple: (x, y_true)
        x, y_true = inputs
        y_pred = self.activation(x)
        # Store y_true and y_pred for loss calculation in `add_loss`
        # Create an instance of WeightedSDRLoss with noisy_signal = x (input)
        loss_fn = WeightedSDRLoss(noisy_signal=x)

        # Add loss computed with y_true and y_pred - Keras sums this during training
        self.add_loss(loss_fn(y_true, y_pred))
        # Forward returns prediction only
        return y_pred


def my_model_function():
    # Return an instance of MyModel with no extra initialization needed
    return MyModel()


def GetInput():
    # Generate sample data that fits the model input requirements
    # The model expects a tuple: (x, y_true)
    # Both with shape (batch_size=1, 4, 1) compatible with example data
    import numpy as np

    batch_size = 1
    # Use random uniform floats for x and y_true as sample inputs
    x = tf.random.uniform((batch_size, 4, 1), dtype=tf.float32)
    y_true = tf.random.uniform((batch_size, 4, 1), dtype=tf.float32)
    return (x, y_true)

