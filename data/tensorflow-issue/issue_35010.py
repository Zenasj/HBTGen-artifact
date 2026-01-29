# tf.random.uniform((256, 10), dtype=tf.int32) ← inferred input shape and type from usage in example loop input

import numpy as np
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, output_dim=1000):
        super(MyModel, self).__init__()
        self.output_dim = output_dim

    def call(self, inputs):
        """
        Calls a py_function inside the forward pass, replicating the original TestLayer behavior
        that caused memory leaks in eager mode. This mock simulates a random output tensor
        shaped (batch_size, output_dim) with dtype float64.
        
        Assumptions and notes:
        - Inputs shape is expected to be (batch_size, feature_dim) with feature_dim=10 as per example.
        - Using tf.py_function here as in original code, which can cause memory leaks in some TF versions.
        """
        # Use tf.py_function to return a random numpy float64 array for each batch example
        batch_embedding = tf.py_function(
            func=self.mock_output,
            inp=[inputs],
            Tout=tf.float64
        )
        # tf.py_function loses static shape info; set shape manually for downstream use
        batch_size = tf.shape(inputs)[0]
        batch_embedding.set_shape([batch_size, self.output_dim])
        return batch_embedding

    def mock_output(self, inputs):
        # inputs is a tf.Tensor; convert to numpy if needed via tf.py_function contract
        # Generate random numpy array with shape (batch_size, output_dim)
        batch_size = inputs.shape[0] if inputs.shape[0] is not None else inputs.shape.as_list()[0]
        # Defensive fallback if shape unknown at call
        if batch_size is None:
            batch_size = 1  # minimal default, but typically known at runtime
        random_output = np.random.random(size=(batch_size, self.output_dim))
        return random_output

def my_model_function():
    # Return an instance of MyModel with default output_dim=1000 as in example
    return MyModel()

def GetInput():
    # Return a random input tensor that matches expected input to MyModel
    # Input shape (batch_size=256, feature_dim=10), dtype int32 (as input consists of random integers 0-100)
    # Matches the example usage: tf.constant(np.random.randint(0,100,(256,10)))
    input_tensor = tf.random.uniform(shape=(256,10), minval=0, maxval=100, dtype=tf.int32)
    return input_tensor

