# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← The input shape is not explicitly stated in the issue;
# we assume a generic 4D tensor typical for Keras Wrappers usage, here (batch, height, width, channels)
import tensorflow as tf


class MyModel(tf.keras.Model):
    """
    This model illustrates a use case for a Keras Wrapper subclass, 
    demonstrating the config mutation issue described.

    We implement a Wrapper subclass (MyWrapper) inside this model, instantiate it 
    with a simple Dense layer, and mimic the from_config usage scenario.
    """

    def __init__(self):
        super().__init__()
        # Internal Wrapper subclass as per the issue example
        class MyWrapper(tf.keras.layers.Wrapper):
            def call(self, inputs, *args, **kwargs):
                return self.layer(inputs, *args, **kwargs)

        self.MyWrapper = MyWrapper

        # A simple wrapped Dense layer, matching example
        self.wrapped_layer = self.MyWrapper(tf.keras.layers.Dense(1))

    def call(self, inputs):
        # Forward simply passes inputs through the wrapped layer
        return self.wrapped_layer(inputs)

    def from_config_behavior(self, config):
        """
        Demonstrates the problem: from_config mutates the input config dict.

        Returns a tuple of (original_config_after_from_config, copy_of_config_before)
        """
        # Copy config for comparison to check later mutation
        config_copy = config.copy()

        # Create new wrapper from config - this call mutates config by popping keys
        new_wrapper = self.MyWrapper.from_config(config)

        # Return mutated original config and the original copy for assertion outside
        return config, config_copy


def my_model_function():
    # Instantiate and return the MyModel instance
    return MyModel()


def GetInput():
    # Return a random 4D input tensor to match typical tf.keras.layers.Wrapper input type
    # The exact shape is not in the issue; assume a batch of 2, with height=4, width=4, channels=3
    return tf.random.uniform((2, 4, 4, 3), dtype=tf.float32)

