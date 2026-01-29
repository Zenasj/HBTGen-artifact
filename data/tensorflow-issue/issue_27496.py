# tf.random.uniform((B, 16), dtype=tf.float32) ← Based on example input shape for the Custom layer with input shape (16,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A custom model that encapsulates the key learnings from the GitHub issue about
    setting 'dynamic=True' in custom layers and the need to implement compute_output_shape.

    This model implements a custom layer similar to the one in the issue comments, which uses
    tf.py_function (showing eager mode use cases) and sets dynamic=True properly by passing it to
    the superclass constructor. It also implements compute_output_shape to avoid NotImplementedError.

    The forward pass runs a py_function and returns a tensor of shape (batch_size, 1).
    """

    class CustomLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            # dynamic must be passed to super().__init__() to properly set the read-only property
            super().__init__(dynamic=True, **kwargs)
            self._outputshape = [None, 1]  # output shape with undefined batch dim

        def build(self, input_shape):
            # No weights in this simple example
            super().build(input_shape)

        def py_function(self, inputs):
            # Example: Sum elements in each sample across last axis, mimicking original example
            # Must use tensorflow ops, but for demonstration use tf.py_function here.
            return tf.reduce_sum(inputs, axis=-1, keepdims=True)

        def call(self, inputs):
            # Use tf.py_function for custom eager execution
            def numpy_func(x):
                # This runs eagerly and sums along axis 1 (same as tf.reduce_sum)
                return x.numpy().sum(axis=1, keepdims=True).astype('float32')

            # Wrap numpy_func in tf.py_function to simulate custom eager behavior
            x = tf.py_function(func=numpy_func, inp=[inputs], Tout=tf.float32)
            x.set_shape(self._outputshape)
            return x

        def compute_output_shape(self, input_shape):
            # Must implement this if dynamic=True to avoid NotImplementedError
            # Output shape: (batch_size, 1)
            batch = input_shape[0] if input_shape else None
            return tf.TensorShape([batch, 1])

    def __init__(self):
        super().__init__()
        self.custom_layer = MyModel.CustomLayer()

    def call(self, inputs):
        # Forward input through the custom layer
        return self.custom_layer(inputs)


def my_model_function():
    # Instantiate and return the model instance
    return MyModel()


def GetInput():
    # Return a random input tensor matching the expected input shape of shape=(batch_size, 16), dtype=float32.
    # Batch size chosen as 4 arbitrarily for demonstration.
    batch_size = 4
    input_shape = (batch_size, 16)
    return tf.random.uniform(input_shape, dtype=tf.float32)

