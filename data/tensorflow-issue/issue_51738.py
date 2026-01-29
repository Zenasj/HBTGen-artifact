# tf.random.uniform((B, 10, 1), dtype=tf.float32) ← Input shape inferred from issue's example: input_shape=(10, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using tf.keras.layers.Dense with input shape (10, 1)
        # To reproduce the scenario in the issue, ensure all layers come from tensorflow.keras
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # inputs expected shape (B, 10, 1)
        # Apply dense layer on last axis
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    # No weights preloaded; model will build on first call
    return MyModel()

def GetInput():
    # Generate a random input tensor matching (B, 10, 1)
    # Using batch size B=4 as a reasonable default for demonstration
    B = 4
    input_shape = (B, 10, 1)
    return tf.random.uniform(input_shape, dtype=tf.float32)

