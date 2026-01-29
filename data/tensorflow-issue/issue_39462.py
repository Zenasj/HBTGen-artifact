# tf.random.normal((10,)) ← Inferred input shape from example: 1D tensor with 10 elements

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    """
    A simple model wrapping a single Dense(1) layer.

    This model emulates the example from the issue demonstrating the 
    ReduceLROnPlateau callback behavior in Keras.

    It also contains a placeholder method 'simulate_reduce_lr_logic' 
    that demonstrates the comparison logic for learning rate adjustment 
    with min_lr, highlighting the precision issue discussed in the issue.

    This method is not part of standard Keras workflow but encapsulates
    the inference from the issue about floating point precision check 
    for learning rate reductions.
    """
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

        # Parameters similar to ReduceLROnPlateau example in the issue
        self.min_lr = 0.001  # minimum learning rate threshold

    def call(self, inputs, training=False):
        return self.dense(inputs)

    def simulate_reduce_lr_logic(self, current_lr):
        """
        Simulates logic similar to ReduceLROnPlateau callback regarding lr reduction.

        Args:
            current_lr (float): current learning rate value.

        Returns:
            bool: True if should reduce learning rate (i.e. current_lr > min_lr considering precision),
                  False otherwise.

        Notes:
            This simulates the bug described: floating-point precision issues in comparisons 
            that might cause reduction logic to run even when lr == min_lr.
        """
        # Due to floating point precision, directly comparing floats is unsafe.
        # Instead, check if current_lr is "significantly" greater than min_lr using a tolerance.
        tolerance = 1e-8
        should_reduce = (current_lr - self.min_lr) > tolerance
        return should_reduce


def my_model_function():
    """
    Instantiate and return the MyModel instance.

    This mimics how a typical keras Model is constructed per the issue example.
    """
    model = MyModel()
    # Build the model by calling it once with a suitable input shape to ensure weights creation.
    dummy_input = tf.random.normal((1, 10))
    model(dummy_input)
    return model


def GetInput():
    """
    Returns a random input tensor that matches the expected input for MyModel.

    The example code uses a 1D vector of length 10 (batch size is flexible).
    This function returns a batch of 1 sample with 10 features, dtype float32.
    """
    return tf.random.normal((1, 10))

