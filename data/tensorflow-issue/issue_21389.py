# tf.random.uniform((B, 3), dtype=tf.float32) and tf.random.uniform((B, 4), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A fused model representing a multi-input Keras model with two inputs:
    one of shape (3,) and one of shape (4,).

    This model concatenates the two inputs and applies a Dense layer
    to produce a single output of shape (5,).

    This is based on the issue context describing multi-input keras models 
    and distribution strategy support and the test cases provided therein.
    """

    def __init__(self):
        super().__init__()
        # Dense layer producing output dim=5 after concatenation of inputs
        self.dense = tf.keras.layers.Dense(5, name='dense')

    def call(self, inputs, training=False):
        """
        Forward pass for the model.

        Parameters:
            inputs: tuple of two tensors:
                - input_a of shape (batch_size, 3)
                - input_b of shape (batch_size, 4)
            training: boolean, unused here but good practice to include

        Returns:
            Tensor of shape (batch_size, 5)
        """
        # Unpack multiple inputs as tuple
        input_a, input_b = inputs

        # Concatenate the inputs along the last axis (feature axis)
        x = tf.keras.layers.concatenate([input_a, input_b])

        # Apply the dense layer
        y = self.dense(x)
        return y


def my_model_function():
    """
    Instantiate and return the MyModel instance.
    """
    model = MyModel()
    # Build the model by calling it once with dummy inputs to create weights
    batch_size = 1  # dummy batch size
    model((tf.zeros((batch_size,3)), tf.zeros((batch_size,4))))
    return model


def GetInput():
    """
    Generate a valid input tuple for MyModel compatible with the expected input.

    Returns:
        tuple of two tensors:
            - tf.Tensor of shape (batch_size, 3), dtype float32
            - tf.Tensor of shape (batch_size, 4), dtype float32
    """
    batch_size = 10  # example batch size to match tests
    input_a = tf.random.uniform((batch_size, 3), dtype=tf.float32)
    input_b = tf.random.uniform((batch_size, 4), dtype=tf.float32)
    return (input_a, input_b)

