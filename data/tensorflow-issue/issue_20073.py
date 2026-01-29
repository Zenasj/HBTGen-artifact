# tf.random.uniform((None, 8, 8, 1), dtype=tf.float32) ← Input shape inferred from minimal reproducible example in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A fused model example combining the insights from the issue discussion.

    This model mimics the minimal CNN that was causing the weight-loading issue when
    InputLayer was separated from the first Conv2D layer.

    The model is constructed with input_shape specified directly on the Conv2D layer,
    avoiding InputLayer. This matches the workaround described.

    The call method forwards input through Conv2D; this simulates the discussed minimal model.
    """
    def __init__(self):
        super().__init__()
        # According to the issue, including InputLayer as separate causes weight loading issue,
        # so here input_shape is specified in first Conv2D layer instead.
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=5,
            input_shape=(8, 8, 1),
            activation=None
        )

    def call(self, inputs):
        return self.conv(inputs)

def my_model_function():
    # Return an initialized instance of MyModel
    # No pretrained weights loaded here, consistent with minimal example from issue
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input shape (batch, height, width, channels)
    # batch dimension is None/variable; here we pick batch = 1 for simplicity
    return tf.random.uniform((1, 8, 8, 1), dtype=tf.float32)

