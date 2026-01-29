# tf.random.uniform((B, 1, 28, 28), dtype=tf.float32) ← Input shape: batch size B, channels_first with 1 channel, 28x28 image

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same layers as in the original Sequential model
        self.conv = tf.keras.layers.Conv2D(
            16, (3, 3), padding='same', data_format='channels_first',
            input_shape=(1, 28, 28))
        self.pool = tf.keras.layers.MaxPooling2D(
            (3, 3), data_format='channels_first')
        # Metrics - as suggested, metrics are only populated after training/eval on data
        # We instantiate these metrics but note they wont show state until update
        self.binary_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.mean_absolute_error = tf.keras.metrics.MeanAbsoluteError()
        # Loss function as in original compile
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        return x

    def compute_metrics(self, y_true, y_pred):
        # Update the metrics states
        self.binary_accuracy.update_state(y_true, y_pred)
        self.mean_absolute_error.update_state(y_true, y_pred)
        # Return current results as a dictionary for convenience
        return {
            'binary_accuracy': self.binary_accuracy.result(),
            'mean_absolute_error': self.mean_absolute_error.result()
        }

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random input matching input_shape (B, C, H, W)
    # Let's arbitrarily choose batch size 4 for demonstration
    B = 4
    # Input dtype float32 for conv2d
    x = tf.random.uniform((B, 1, 28, 28), dtype=tf.float32)
    # As this is a binary classification example, generate dummy binary labels
    # Shape should correspond to output shape or target label shape accordingly
    # For simplicity, create binary labels that match batch size and possible output dim
    # Since model output shape is not a classifier output here but feature map,
    # make dummy labels compatible with flatten + dense or you can just return random labels.
    # Here as a stub, just return a random binary label tensor with shape (B, 1).
    y = tf.random.uniform([B, 1], maxval=2, dtype=tf.int32)
    y = tf.cast(y, tf.float32)
    # Return inputs and labels as tuple to use with compute_metrics function
    return x, y

