# tf.random.uniform((1, 257, 257, 3), dtype=tf.float32)  ← inferred input shape and dtype based on example from the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # DepthwiseConv2D layer with dilation_rate=2 and no bias as per example in the issue.
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=64,
            dilation_rate=2,
            use_bias=False,
            padding='same'  # padding to preserve input spatial dimensions
        )
        # The Lambda layer mimics the workaround mentioned in the issue:
        # Adding a no-op lambda that adds 0.0 to the output,
        # which helped TOCO identify dilated conv pattern correctly.
        self.workaround = tf.keras.layers.Lambda(lambda x: x + 0.0)

    def call(self, inputs, training=False):
        x = self.depthwise_conv(inputs)
        x = self.workaround(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching shape (batch=1, height=257, width=257, channels=3)
    # dtype float32 matching typical image input for DepthwiseConv2D
    return tf.random.uniform((1, 257, 257, 3), dtype=tf.float32)

