# tf.random.uniform((B, 1), dtype=tf.float32) ← input shape inferred from Dense layer input_shape=(1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model matching the reported model:
        # A single Dense layer with 1 output unit, input_shape=(1,)
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate random input tensor matching shape (batch_size, 1)
    # Use batch size 8 as a reasonable default
    return tf.random.uniform((8, 1), dtype=tf.float32)

