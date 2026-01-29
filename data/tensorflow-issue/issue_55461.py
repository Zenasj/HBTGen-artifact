# tf.random.uniform((1, 1, 2), dtype=tf.float32), tf.random.uniform((1, 3, 1), dtype=tf.float32)

import tensorflow as tf
from tensorflow_graphics.math.optimizer import levenberg_marquardt

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Define the residual functions for Levenberg-Marquardt optimizer
        # These are simple example residuals used in the original issue:
        # f1(x,y) = x + y, f2(x,y) = x * y
        self.f1 = lambda x, y: x + y
        self.f2 = lambda x, y: x * y

    def call(self, inputs):
        """
        Forward pass of the model using Levenberg-Marquardt minimization.
        
        Args:
          inputs: list of two tensors [x, y]
            - x shape: (batch, 1, 2)
            - y shape: (batch, 3, 1)
        
        Returns:
          List of tensors representing residuals after minimization (r1, r2).
        """
        x, y = inputs

        # Run Levenberg-Marquardt minimization with these residuals and variables
        _, (r1, r2) = levenberg_marquardt.minimize(
            residuals=(self.f1, self.f2),
            variables=[x, y],
            max_iterations=10,
        )
        
        # Return the residual tensors as output
        return [r1, r2]

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two inputs compatible with MyModel call:
    # - x: shape (1, 1, 2), dtype float32
    # - y: shape (1, 3, 1), dtype float32
    x = tf.random.uniform((1, 1, 2), dtype=tf.float32)
    y = tf.random.uniform((1, 3, 1), dtype=tf.float32)
    return [x, y]

