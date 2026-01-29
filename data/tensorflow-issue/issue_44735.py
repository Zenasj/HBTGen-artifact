# tf.random.uniform((1, 4, 2), dtype=tf.float32) ← inferred input shapes from example: batch_size=1, shape=(4, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable weights in this model; just a BatchMatMul (matmul with transpose_b=True)
        # using inputs shape [1, 4, 2], multiplying input1 @ input2^T on last dims
        # We'll keep it simple with direct matmul in call.
    
    def call(self, inputs):
        # inputs is a tuple/list of two tensors each of shape [1, 4, 2]
        input1, input2 = inputs
        # Perform the batch matmul with transpose_b=True
        # That means multiplying input1 with the transpose of input2 on the last dimension
        # Shapes:
        # input1: [1, 4, 2]
        # input2: [1, 4, 2]
        # tf.linalg.matmul with transpose_b=True on these shapes does batch matmul over [1] batch dim:
        # Output shape: [1, 4, 4] (because matrix multiply (4x2) @ (2x4))
        output = tf.linalg.matmul(input1, input2, transpose_b=True)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two random tensors matching input shape [1, 4, 2], dtype float32
    # Using tf.random.uniform as per requirement
    input_shape = (1, 4, 2)
    input1 = tf.random.uniform(input_shape, dtype=tf.float32)
    input2 = tf.random.uniform(input_shape, dtype=tf.float32)
    return (input1, input2)

