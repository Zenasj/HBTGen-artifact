# tf.random.uniform((B, 16), dtype=tf.float32) ← Input shape inferred from `Input(shape=(16,))` in original Keras functional example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with 16 units, no activation specified (linear by default)
        self.dense = tf.keras.layers.Dense(16)
        
    def call(self, inputs):
        # Forward pass
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The original input shape was (160, 16) for training, but batch size can be arbitrary here.
    # Use batch size = 32 as in example; dtype float32 for Dense layer compatibility.
    return tf.random.uniform((32, 16), dtype=tf.float32)

