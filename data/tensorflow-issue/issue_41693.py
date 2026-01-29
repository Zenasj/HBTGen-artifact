# tf.random.uniform((100, 10), dtype=tf.float32) ← inferred input shape from provided issue code example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def build(self, batch_input_shape):
        # Create a Dense layer as output_layer during build
        self.output_layer = tf.keras.layers.Dense(1)
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        # Use a callable loss instead of a constant tensor loss to avoid InaccessibleTensorError
        # This resolves the issue described in the GitHub issue where add_loss(1.0) causes errors
        self.add_loss(lambda: 1.0)
        return self.output_layer(inputs)

def my_model_function():
    # Return an instance of MyModel, ready for compilation and training
    return MyModel()

def GetInput():
    # Generate a random input tensor matching expected input shape (batch_size=100, features=10)
    # dtype is float32 by default for tf.random.uniform
    return tf.random.uniform((100, 10))

