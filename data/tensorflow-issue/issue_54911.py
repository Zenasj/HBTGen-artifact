# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32) ← inferred input shape from issue example

import tensorflow as tf

class StdConv(tf.keras.layers.Conv2D):
    """Weight Standardization Conv2D.

    See https://arxiv.org/pdf/1903.10520v1.pdf.

    This layer standardizes its kernel weights during call before convolution.
    """

    def _standardize_wts(self, wts):
        # Compute mean and variance across height, width, and input channel dims,
        # keep dims for broadcasting.
        wts_mean = tf.math.reduce_mean(wts, axis=(0, 1, 2), keepdims=True)
        wts_var = tf.math.reduce_variance(wts, axis=(0, 1, 2), keepdims=True)
        # Standardize weights with numerical stabilization.
        return (wts - wts_mean) / tf.math.sqrt(wts_var + 1e-5)

    def call(self, inputs):
        # Standardize the conv kernel weights and assign them back.
        # NOTE: This introduces a variable assignment inside call, which may cause issues
        # with graph tracing or TF-TRT conversion as described in the issue.
        standardized_wts = self._standardize_wts(self.kernel)
        self.kernel.assign(standardized_wts)
        # Perform convolution with standardized weights.
        return super().call(inputs)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single StdConv layer with 64 filters and 7x7 kernel, padding same.
        self.root = StdConv(
            filters=64,
            kernel_size=7,
            padding="same",
            name="conv_root",
        )

    def call(self, inputs, training=True):
        # Forward pass through the standardized convolution.
        return self.root(inputs, training=training)


def my_model_function():
    # Instantiate and return an instance of MyModel.
    return MyModel()


def GetInput():
    # Return a random input tensor of shape (1, 256, 256, 3), dtype float32
    # matching the original example shape.
    return tf.random.uniform(shape=(1, 256, 256, 3), dtype=tf.float32)

