# tf.random.uniform((B,), dtype=tf.float32) ← The input shape is [batch_size], scalar inputs per batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Here we create a simple model that just reduces sum of scalar inputs
        # This replicates the ToyModel behavior from the issue.
        # No layers needed since the model just sums scalar inputs.
    
    def call(self, inputs):
        # inputs shape expected: [batch_size] (scalar per batch element)
        # Just return the sum over batch for demonstration
        # We use tf.reduce_sum to output a scalar per batch (or single scalar)
        # This simple operation exposes the input shape in call for debugging
        tf.print("Input shape is", tf.shape(inputs))
        return tf.reduce_sum(inputs, axis=-1)

def my_model_function():
    # Instantiate and return the MyModel model
    # No weights or initialization needed beyond constructor here
    return MyModel()

def GetInput():
    # Return a batch of scalar inputs consistent with MyModel expectations
    # For example, batch size 8, scalar inputs per batch element (shape [8])
    batch_size = 8
    # Generate a random float32 tensor of shape [batch_size]
    return tf.random.uniform((batch_size,), dtype=tf.float32)

