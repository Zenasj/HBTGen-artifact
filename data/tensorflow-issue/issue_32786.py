# tf.random.uniform((1, 2), dtype=tf.float32) ← Input shape inferred from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use a custom layer similar to the MyLayer in the issue,
        # demonstrating the build/input_shape behavior.
        self.my_layer = MyLayer()

    def call(self, inputs):
        # Pass inputs through the custom layer unchanged.
        return self.my_layer(inputs)

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.expected_shape = (1, 2)  # Known input shape from the issue example

    def build(self, input_shape):
        # Print and assert the input_shape matches expected shape during build
        # This reproduces the original user's concern about correct build input shape
        # In real scenario, use build to create weights, here just demonstrate shape handling.
        print("Building with input_shape:", input_shape)
        assert tuple(input_shape) == self.expected_shape, \
            f"Expected input shape {self.expected_shape}, got {input_shape}"
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        return inputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # From the issue's minimal example: shape = (1, 2)
    return tf.random.uniform((1, 2), dtype=tf.float32)

