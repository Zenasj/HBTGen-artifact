# tf.random.uniform((B, 1), dtype=tf.float32), tf.random.uniform((B, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights in this simple CustomLayer logic encapsulated here

    def call(self, inputs):
        # inputs expected to be a list/tuple of two tensors with shape (batch_size, 1)
        # We check only the first dimension (batch dimension) of each input tensor
        input_a, input_b = inputs
        # Assert batch size equality with tf.assert_equal on batch dimension
        batch_size_a = tf.shape(input_a)[0]
        batch_size_b = tf.shape(input_b)[0]
        
        assert_same_batch_size = tf.debugging.assert_equal(
            batch_size_a,
            batch_size_b,
            message="inputs do not have equal batch_size"
        )
        # Use control_dependencies to ensure the assert runs before further ops
        with tf.control_dependencies([assert_same_batch_size]):
            # Reshape input_a to (batch_size, 3)
            # This will cause an error if batch_size*3 != total elements in input_a
            # We assume input_a originally has shape (batch_size, 1)
            # So this reshape only works if batch_size*3 == batch_size*1 => 3=1 no, so reshape is intentionally breaking for demonstration
            # To avoid breaking during normal input, let’s create a tensor of shape (batch_size, 3) from input_a via tiling or something similar
            # But since the original code wants this reshape to break if inputs are malformed, we replicate the same behavior:
            reshaped = tf.reshape(input_a, (batch_size_a, 3))
            return reshaped

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tuple of two input tensors each with batch dimension and feature dim=1
    # Using batch size 10 but second tensor has different batch size 15, to trigger assert
    input_a = tf.random.uniform((10, 1), dtype=tf.float32)
    input_b = tf.random.uniform((10, 1), dtype=tf.float32)
    # Note: Using matching batch_size 10 and 10 here so the assert will pass for normal input
    # To test error scenario, caller can provide mismatched batch sizes
    
    # For demonstration, return matching batch sizes to not break here by default
    return (input_a, input_b)

