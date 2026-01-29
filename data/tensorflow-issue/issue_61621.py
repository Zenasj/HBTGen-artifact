# tf.random.uniform((1, 1, 2, 2, 3), dtype=tf.float32) ← Input shape from issue reproduction snippet

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize ZeroPadding3D layer with large padding causing overflow bug
        # The padding is passed as a single large integer as in the issue (padding=1610612736)
        # Note: This large padding triggers the overflow bug when this model is run.
        self.zero_padding = tf.keras.layers.ZeroPadding3D(padding=1610612736)

    def call(self, inputs):
        # The forward pass applies the ZeroPadding3D layer
        # This is expected to raise an overflow error due to large padding
        return self.zero_padding(inputs)

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a tensor matching the expected input shape for MyModel
    # Shape inferred from the issue reproduction code: [1, 1, 2, 2, 3], dtype float32
    return tf.random.uniform((1, 1, 2, 2, 3), dtype=tf.float32)

