# This example illustrates a model with multiple signatures (methods) 
# that take no meaningful input, illustrating the issue of concrete functions with no inputs.
# Input shape is empty (no input tensors)
# tf.random.uniform(())  # no input shape since functions take no inputs

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    @tf.function
    def test_1(self):
        # Returns a constant integer; no input arguments
        return tf.constant(123)

    @tf.function
    def test_2(self):
        # Returns another constant integer; no input arguments
        return tf.constant(456)

    # For demonstration, a method showing signature with a dummy input to workaround known issues,
    # but note that this breaks compatibility with older signatures as discussed.
    @tf.function
    def test_dummy(self, dummy):
        # Just returns dummy (ignored)
        return dummy


def my_model_function():
    # Return an instance of MyModel (no special weights or initialization needed)
    return MyModel()


def GetInput():
    # Since the main functions have no inputs, we return an empty tuple to match their expected input.
    # For test_dummy (not usually called), input would be a scalar tensor.
    # But to be safe, the model's call expects no arguments, so return no input.
    return ()

