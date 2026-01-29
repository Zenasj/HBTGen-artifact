# tf.random.uniform((B, 16), dtype=tf.float32) ← Input shape inferred from input_shape=(16,) in Dense layer

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple model corresponding to the example Dense layer with input shape (16,)
        self.dense = tf.keras.layers.Dense(4, input_shape=(16,))
    
    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape (batch_size, 16)
    # Assume batch_size=8 as a reasonable default
    batch_size = 8
    return tf.random.uniform((batch_size, 16), dtype=tf.float32)

