# tf.random.uniform((10, 1), dtype=tf.float32) ← inferred input shape from dataset batch size and input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code uses a simple sequential model with one Dense layer input_shape=(1,)
        # Here we replicate that with a single Dense layer.
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape expected by MyModel
    # The example uses batches of size 10, each element shape (1,)
    # Use float32 dtype, consistent with typical TF defaults
    return tf.random.uniform((10, 1), dtype=tf.float32)

