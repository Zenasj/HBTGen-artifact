# tf.random.uniform((1, 1, 10, 1), dtype=tf.float32) ← input shape assumed for demonstration, width > rlim test

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use a zero-valued 1x1 Conv2D filter with 1 input and output channel to mimic the OP raw conv2d filter.
        self.conv_filter = tf.zeros([1, 1, 1, 1], dtype=tf.float32)
        self.strides = [1, 2, 2, 1]
        self.padding = "SAME"

    def call(self, inputs):
        # Assume inputs is a tuple or list of (x, rlim)
        x, rlim = inputs

        # The original bug context is that slicing x[:, :, :rlim, :] when rlim > width dimension
        # is allowed in Python (slice end clipped to max dimension size), but causes TFLite crash.
        # We reproduce the safe behavior with explicit min to avoid unsafe slicing in TFLite.

        # Get width dimension of x (assumed NHWC)
        width = tf.shape(x)[2]

        # Compute safe slice end index: min(width, rlim)
        safe_rlim = tf.minimum(width, rlim)

        # Slice the input tensor safely
        x_sliced = x[:, :, :safe_rlim, :]

        # Perform Conv2D with the zero filter and given strides/padding.
        # Using tf.raw_ops.Conv2D directly to match original code.
        out = tf.raw_ops.Conv2D(
            input=x_sliced,
            filter=self.conv_filter,
            strides=self.strides,
            padding=self.padding
        )
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Provide a sample input tensor and rlim tensor
    # Shape chosen as (1, 1, 10, 1) for clarity: batch=1, height=1, width=10, channels=1
    # rlim chosen > width to test safe clipping behavior
    x = tf.random.uniform(shape=(1, 1, 10, 1), dtype=tf.float32)
    rlim = tf.constant(2**31, dtype=tf.int64)  # Large rlim to trigger clipping logic
    return (x, rlim)

