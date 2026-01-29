# tf.random.uniform((B, 0), dtype=tf.float32) ← inferred input shape from the issue: batch size B, feature dimension 0

import tensorflow as tf

class CustomRematLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # remat wraps intermediate_function with recomputation for memory efficiency
        self.remat_function = tf.keras.remat(self.intermediate_function)

    def intermediate_function(self, x):
        # Simple identity scaling as in original example
        return x * 1.0

    def call(self, inputs):
        return self.remat_function(inputs)


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_remat = CustomRematLayer()
        # Since input feature dim is 0, use a Dense layer with input shape compatible with (None, 0)
        # TensorFlow Dense layer requires input dim >0, but 0 dimension input is unusual.
        # To handle this, create a Dense layer with units=1 but input shape (None,0).
        # Dense layer with input shape (None,0) has no weights, acts as bias-only layer.
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.custom_remat(inputs)
        output = self.dense(x)
        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor with batch size 32 and feature dimensions 0 as per the issue input
    # This mimics np.random.randn(32, 0)
    # TensorFlow does not support shape with zero dimensions in tf.random.uniform directly,
    # but shape (32, 0) is valid and produces an empty tensor.
    return tf.random.uniform((32, 0), dtype=tf.float32)

