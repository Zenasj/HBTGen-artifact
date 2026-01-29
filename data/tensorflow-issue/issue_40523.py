# tf.random.uniform((batch_size, 100, 12), dtype=tf.float32)  ← inferred input shape from original reported issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple dense layer applied on last dimension (12) to output a scalar per timestep
        # This replicates the example: outputs = Dense(1)(inputs)
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 100, 12)
        # Apply dense layer to last dim -> output shape: (batch_size, 100, 1)
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the input shape expected by MyModel:
    # (batch_size, 100, 12) float32
    # Let's pick batch_size=32 as default to match original batch_size in the issue.
    batch_size = 32
    return tf.random.uniform((batch_size, 100, 12), dtype=tf.float32)

