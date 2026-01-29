# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred from keras.Input((5)) in example

import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    """Combine multiple activations weighted by learnable variables (as per original example)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        # Returning empty dict as original example has no config parameters
        return {}

    def build(self, input_shape):
        # No weights or parameters, so nothing to build here
        super().build(input_shape)

    def call(self, inputs):
        # Identity mapping as per original example
        return inputs

class MyModel(tf.keras.Model):
    """
    Model equivalent to the example:
    Input shape: (None, 5)
    Consists of a single CustomLayer identity layer.
    """
    def __init__(self):
        super().__init__()
        self.custom_layer = CustomLayer()

    def call(self, inputs):
        return self.custom_layer(inputs)

def my_model_function():
    # Returns an instance of MyModel as required
    return MyModel()

def GetInput():
    # Return a random input matching expected input shape (batch size 1, 5 features)
    # Using float32 dtype as is standard for TF models
    return tf.random.uniform((1, 5), dtype=tf.float32)

