# tf.random.uniform((2, 7, 1, 1), dtype=tf.float64), minval=0, maxval=0 (note: maxval same as minval leads to all zeros)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use relu Activation layer as in the issue
        self.activation = tf.keras.layers.Activation("relu")

    def call(self, inputs, training=False):
        """
        Forward pass applies ReLU activation to the inputs.
        """
        return self.activation(inputs)


def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected input shape of MyModel.
    The original bug report used shape [2, 7, 1, 1] with all zeros (minval=maxval=0),
    dtype=tf.float64.
    
    Using the same for precise replication.
    """
    # The original code uses uniform with minval=0, maxval=0 (all zeros).
    # To precisely replicate the original bug case, we return a zero tensor.
    input_tensor = tf.random.uniform(
        shape=(2, 7, 1, 1),
        minval=0,
        maxval=0,
        dtype=tf.float64
    )
    return input_tensor

