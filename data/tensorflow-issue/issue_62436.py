# tf.random.uniform((2, 1, 1, 1), dtype=tf.float64)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Alpha value from original issue
        self.alpha = 0.2
        # Use LeakyReLU layer with dtype float64 for numerical precision in gradient testing
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=self.alpha, dtype=tf.float64)
    
    def call(self, inputs):
        # Forward pass through LeakyReLU layer
        return self.leaky_relu(inputs)
    
    def compute_gradient_comparison(self, inputs):
        """
        Compute both theoretical and numerical gradients of the layer with respect to inputs
        and return them to compare the bug reported on tf.test.compute_gradient.

        Returns a tuple of:
        - theoretical gradient tensor
        - numerical gradient tensor
        """
        theoretical, numerical = tf.test.compute_gradient(self.leaky_relu, [inputs])
        # Return the first gradient from each tuple (only one input tensor)
        return theoretical[0], numerical[0]

def my_model_function():
    # Return an initialized instance of MyModel
    return MyModel()

def GetInput():
    # Based on the original issue, input shape is [2,1,1,1] with dtype float64
    # Avoid minval=maxval=0 to prevent zero tensor causing wrong gradients
    # Use non-zero range as recommended in the issue discussion
    return tf.random.uniform(
        shape=(2, 1, 1, 1), minval=-1.0, maxval=1.0, dtype=tf.float64
    )

