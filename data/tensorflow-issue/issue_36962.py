# tf.random.uniform((B, 5), dtype=tf.float32)  # Input shape: (batch_size, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # CustomLayer implementation that correctly assigns a name to its weight.
        # This addresses the original issue where add_weight without name causes saving errors.
        self.custom_layer = self.CustomLayer()

    class CustomLayer(tf.keras.layers.Layer):
        def build(self, input_shape):
            # IMPORTANT: Providing a name to avoid the error described in the issue.
            self.w = self.add_weight(
                shape=(input_shape[-1], 1),
                initializer='random_normal',
                trainable=True,
                name='weight')  # Name must be provided to avoid saving issues.

        def call(self, x):
            return tf.matmul(x, self.w)

    def call(self, inputs):
        # Forward pass through the custom layer
        return self.custom_layer(inputs)

def my_model_function():
    # Return an instance of the model.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected shape (batch_size, 5).
    # Use batch size = 4 as a reasonable default example.
    return tf.random.uniform((4, 5), dtype=tf.float32)

