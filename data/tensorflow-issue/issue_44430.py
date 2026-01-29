# tf.random.uniform((B,), dtype=tf.float32) ← Input shape inferred as 1D vector of floats, since original example used np.arange(10.)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = tf.keras.layers.Dense(1)

    @tf.function  # This decoration is recommended to avoid AutoGraph errors during saving
    def call(self, inputs):
        x = inputs
        # In the original issue, iterating over a tf.Tensor with tf.range caused save() to fail.
        # Using tf.function and tf.range inside the call is correct, but needs @tf.function decoration.
        # The original loop: for i in tf.range(2): x = self.layer(x)
        # This runs the Dense layer twice sequentially.
        for _ in tf.range(2):
            x = self.layer(x)
        return x

def my_model_function():
    # Return a new instance of the model with initialized weights
    return MyModel()

def GetInput():
    # Return a random input tensor shape that works with MyModel.
    # The model expects a 1D vector (per the repro code np.arange(10.)),
    # so shape (10,) and dtype float32 is suitable.
    return tf.random.uniform((10,), dtype=tf.float32)

