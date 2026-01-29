# tf.random.uniform((B, 1), dtype=tf.float32) ← Assuming batch dimension B and input shape (1,)

import tensorflow as tf
from tensorflow.keras import layers

class VariableLayer(layers.Layer):
    """A Keras Layer that holds a tf.Variable and returns it on call.
    This layer takes no inputs (empty input) and just returns its variable.
    Equivalent to the custom Variable Layer from the issue."""
    def __init__(self, initial_value, **kwargs):
        super(VariableLayer, self).__init__(**kwargs)
        # Create a trainable variable initialized to initial_value
        self.var = tf.Variable(initial_value, trainable=True)

    def call(self, inputs):
        # inputs is expected to be an empty list or tensor; not used here
        # Return the variable as a tensor broadcasted as needed
        return self.var

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Input shape is (None, 1), batch dimension is None
        # Regular input layer
        self.input_layer = layers.InputLayer(input_shape=(1,))
        # Our custom variable layer that returns a variable. No inputs expected.
        self.variable_layer = VariableLayer([1.0])
        # Add layer to combine input and variable
        self.add_layer = layers.Add()

    def call(self, inputs):
        """
        inputs: a tensor of shape (B, 1)
        Returns: tensor output = input + variable
        """
        # inputs comes directly from input
        # variable_layer takes an empty input list [] as per the original example
        var_output = self.variable_layer([])
        # Add the input tensor and variable tensor
        # var_output shape is scalar, will broadcast over batch dimension
        return self.add_layer([inputs, var_output])

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (B, 1)
    # Assuming batch size 4 for demonstration, float32 as typical for Keras models
    B = 4
    return tf.random.uniform((B, 1), dtype=tf.float32)

