# tf.random.uniform((B, 5, 256), dtype=tf.float32) ← Input shape is (batch_size, 5, 256) as per the original issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use tf.keras.layers.Dense as recommended to avoid issues with tf.layers.dense
        # units=513, activation='relu' as in original code
        self.dense_layer = tf.keras.layers.Dense(units=513, activation='relu')
        
    def call(self, inputs):
        # Apply the dense layer to inputs
        # Inputs shape: (batch_size, 5, 256)
        # Dense applies last dimension transformation: 256 -> 513
        return self.dense_layer(inputs)

def my_model_function():
    # Return an instance of MyModel, weights initialized by default
    return MyModel()

def GetInput():
    # Generate a random float32 tensor matching input shape (batch_size, 5, 256)
    # Batch size use 16 as a reasonable default based on original usage
    return tf.random.uniform((16, 5, 256), dtype=tf.float32)

