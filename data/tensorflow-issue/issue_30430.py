# tf.random.uniform((batch_size, 1000, 512), dtype=tf.float32) ← inferred input shape from issue code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # From the issue example: Input shape is (1000, 512)
        # The model uses a GRU with units = shape[-1] // 2 = 512 // 2 = 256, return_sequences=True
        self.gru = tf.keras.layers.GRU(256, return_sequences=True)
    
    def call(self, inputs, training=False):
        # Forward pass: run GRU on inputs
        return self.gru(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel:
    # The original example uses ones with shape (batch, 1000, 512)
    # We'll produce a float32 tensor with batch size 4 as a reasonable default
    batch_size = 4
    H, W = 1000, 512
    # Use uniform random data for diversity, dtype float32
    return tf.random.uniform((batch_size, H, W), dtype=tf.float32)

