# tf.random.uniform((1,), dtype=tf.float32) ← Input shape inferred from issue example: 1D tensor with unknown batch size 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No layers needed; the model just applies tf.reverse with empty axis list
        # which is effectively a no-op but triggers the original reported issue.
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def call(self, x):
        # Apply tf.reverse with axis=[] which is a no-op by definition
        # Included to reproduce original intent from the issue
        return tf.reverse(x, axis=[])

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tensor matching the input signature expected by MyModel
    # Shape is [None], so generate a 1D float32 tensor with length 1 for simplicity
    return tf.random.uniform(shape=[1], dtype=tf.float32)

