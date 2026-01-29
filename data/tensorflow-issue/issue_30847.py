# tf.random.uniform((256, 20), dtype=tf.int32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The sample model from the issue uses a Dense layer on numeric inputs
        self.dense = tf.keras.layers.Dense(1, name='output')

    def call(self, inputs):
        """
        inputs: a tuple or dict of inputs, containing
          - string input tensor of shape (batch_size, 1) (not used in computation)
          - numeric input tensor of shape (batch_size, 4) (used by Dense layer)
        The model ignores string input for computation but requires it in the dataset.
        """
        # Assume inputs is a tuple or list: (string_input, numeric_input)
        # Just call dense on numeric_input
        # Ignore string_input aside from passing as part of inputs.
        string_input, numeric_input = inputs
        output = self.dense(numeric_input)
        return output

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple matching the expected inputs:
    # string input shape: (batch_size, 1) of dtype string
    # numeric input shape: (batch_size, 4) of dtype float32
    batch_size = 6  # Arbitrary batch size matching multiples of replicas (e.g., 2 replicas * 3 =6)
    string_input = tf.constant([["This is a string"]] * batch_size, dtype=tf.string)
    numeric_input = tf.random.uniform((batch_size, 4), dtype=tf.float32)
    return (string_input, numeric_input)

