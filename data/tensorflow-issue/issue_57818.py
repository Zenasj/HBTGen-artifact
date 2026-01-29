# tf.random.uniform((1,), dtype=tf.float64) ← input shape inferred from example input tf.constant([3.12363398], dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Constant input value used in original example, dtype float64
        self.const = tf.constant([-10.43154963850975037], dtype=tf.float64)

    @tf.function
    def call(self, x):
        # Apply leaky relu to the constant (alpha=0.1)
        c = tf.raw_ops.LeakyRelu(features=self.const, alpha=0.1)
        # Add the result of leakyrelu(const) to input x
        out = tf.add(c, x)
        return out

def my_model_function():
    # Return an instance of MyModel, no additional initialization required
    return MyModel()

def GetInput():
    # Produce a random input tensor matching the example shape and dtype
    
    # The example input used a 1-D tensor with a single float64 element.
    # We will generate a similar tensor with one element, dtype float64.
    input_tensor = tf.random.uniform(
        shape=(1,), minval=-20.0, maxval=20.0, dtype=tf.float64
    )
    return input_tensor

