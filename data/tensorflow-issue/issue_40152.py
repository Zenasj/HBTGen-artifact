# tf.random.uniform((B, 32, 32), dtype=tf.float32) ← inferred input shape from the minimal example (batch size B is dynamic)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The minimal model from the issue consists of a single ELU activation layer
        # The input shape is (32, 32), so the input tensor is expected to have shape (B, 32, 32)
        self.elu = tf.keras.layers.ELU()

    def call(self, inputs):
        # inputs shape: (B, 32, 32)
        # ELU activation applies element-wise activation
        return self.elu(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor matching the input expected by MyModel
    # batch size = 1 for default usage; shape (1, 32, 32)
    # Using uniform distribution between -1 and 1 as ELU handles negative and positive values
    return tf.random.uniform((1, 32, 32), minval=-1.0, maxval=1.0, dtype=tf.float32)

