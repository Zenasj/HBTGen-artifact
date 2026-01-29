# tf.random.uniform((1, 5000), dtype=tf.float32) ← Input shape inferred from issue example: batch=1, features=5000

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a single dense layer matching the example: units=5000, input shape=(5000,)
        self.dense = tf.keras.layers.Dense(units=5000)

    def call(self, inputs, training=False):
        # Forward pass through dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input matching the expected shape (1, 5000) and float32 dtype
    return tf.random.uniform((1, 5000), dtype=tf.float32)

