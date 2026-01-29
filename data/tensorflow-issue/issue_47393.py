# tf.random.uniform((), dtype=tf.float32) ← Input is a scalar tensor as the issue revolves around scalar vs non-scalar EagerTensors

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use sigmoid activation as per the example in the issue
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        # inputs expected to be a scalar or tensor
        # Apply sigmoid activation
        activated = self.sigmoid(inputs)
        
        # The core issue described: len() fails on scalars, so implement a safe length check
        # Return a dictionary with the activated tensor and length check result for demonstration
        length = self.safe_len(activated)
        return {
            "activated": activated,
            "length_or_error_msg": length
        }

    def safe_len(self, tensor):
        # Attempt to get len(tensor) safely for scalar or non-scalar tensor
        # Scalars have shape () and raise TypeError on len()
        try:
            # len() works if tensor shape has dimension(s)
            return len(tensor)
        except TypeError:
            # For scalars, len() is not valid; return a descriptive message
            return "Scalar tensor has no len()"

def my_model_function():
    # Returns an instance of MyModel 
    return MyModel()

def GetInput():
    # Based on the issue, input can be scalar float32 tensor, e.g., tf.constant(3.0)
    # Matching the example from the issue: scalar tensor of dtype float32
    return tf.constant(3.0, dtype=tf.float32)

