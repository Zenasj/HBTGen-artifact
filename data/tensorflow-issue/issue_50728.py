# tf.random.uniform((3, 10, 3), dtype=tf.float32) ← Input shape inferred from example in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    Model reproducing the reported behavior of a Conv1D-derived layer that modifies output shape.

    Implements a custom Conv1D layer subclass that performs a Conv1D operation followed by a reshape
    which flattens the last two dimensions into one. This simulates subpixel convolution (Conv1D + reshape).
    """

    def __init__(self):
        super().__init__()
        # Create a Conv1D layer with 3 filters and kernel size 3, padding='same' as in example
        self.conv = tf.keras.layers.Conv1D(filters=3, kernel_size=3, padding="same")

    def compute_output_shape(self, input_shape):
        """
        Override compute_output_shape to flatten the last two dimensions after convolution,
        similar to the example in the issue.
        """
        # Use Conv1D's compute_output_shape
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        # conv_output_shape is expected to be (batch_size, length, channels)
        batch_size, length, channels = conv_output_shape
        # Flatten length and channels into one dimension
        return (batch_size, length * channels)

    @tf.function
    def call(self, inputs):
        """
        Perform Conv1D followed by flattening the spatial and channel dimensions into one dimension,
        consistent with subpixel convolution/reshaping described in the issue.
        """
        # Perform convolution
        conv_out = self.conv(inputs)
        # Reshape: keep batch dimension; flatten the rest
        batch_size = tf.shape(conv_out)[0]
        reshaped = tf.reshape(conv_out, (batch_size, -1))
        return reshaped

def my_model_function():
    """
    Return an instance of MyModel with Conv1D + reshape functionality.
    """
    return MyModel()

def GetInput():
    """
    Return a random tensor input matching the input expected by MyModel.
    Input shape [batch_size=3, length=10, channels=3] (based on issue reproduction code).
    """
    return tf.random.uniform((3, 10, 3), dtype=tf.float32)

