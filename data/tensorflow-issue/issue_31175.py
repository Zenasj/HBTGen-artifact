# tf.random.uniform((128,), dtype=tf.int32) ← Input is a 1D tensor with 128 elements, as from dataset tf.data.Dataset.from_tensor_slices(list(range(128)))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No layers needed for a simple step doubling function
        # Using a method step that doubles input

    def step(self, X):
        # Simply multiply input tensor elements by 2
        return X * 2

    def call(self, inputs):
        # For demonstration, this simulates the epoch loop behavior from the issue
        # inputs: a tensor representing a batch (or sequence) of input values
        # We simulate the autograph loop by iterating over inputs and applying step
        # In practice, passing the method as an argument leads to retracing issues.
        # Here, we demonstrate safe usage by calling self.step directly inside call.
        outputs = []
        # Using tf.map_fn for vectorized behavior and to keep TF graph-friendly
        outputs = tf.map_fn(self.step, inputs)
        return outputs

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return random input tensor to simulate dataset from 0 to 127
    # Match the input shape expected: one-dimensional tensor of 128 elements
    # Using tf.random.uniform to generate int32 tensor in range [0,128)
    return tf.random.uniform(
        shape=(128,),
        minval=0,
        maxval=128,
        dtype=tf.int32,
    )

