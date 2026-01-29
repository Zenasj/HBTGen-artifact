# tf.random.uniform((2, 3, 4), dtype=tf.float32) ← inferred input shape and dtype from issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, alpha=14.5):
        super().__init__()
        # Initialize an ELU layer with the specified alpha and float64 dtype as in the issue example
        self.elu_layer = tf.keras.layers.ELU(alpha=alpha, dtype=tf.float64)
        self.alpha = alpha

    def call(self, inputs):
        # Forward pass through ELU layer
        return self.elu_layer(inputs)

    def compute_theoretical_numerical_gradients(self, inputs):
        """
        Compute theoretical and numerical gradients of ELU layer for the given inputs.
        Returns tuples of gradients as in the original test.
        """
        theoretical, numerical = tf.test.compute_gradient(self.elu_layer, [inputs])
        return theoretical, numerical

def my_model_function():
    # Return an instance of MyModel initialized with default alpha=14.5
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches ELU input shape from the example: [2,3,4]
    # Use float64 since ELU layer dtype is float64, but original input in code was tf.float32 - 
    # we'll cast to float64 for compatibility to avoid dtype issues.
    # Values chosen between -0.01 and 0.01 to cover inputs around zero where gradient discrepancy arises
    return tf.random.uniform(shape=(2, 3, 4), minval=-0.01, maxval=0.01, dtype=tf.float64)

