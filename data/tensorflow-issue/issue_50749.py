# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape from the example input of shape (100,) reshaped as (B,1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(10)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(inputs)  # Note: original code used input again, maybe a bug but preserved here
        # Return the output tensor
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input consistent with input shape (B,1)
    # Assume batch size 32 for generality
    return tf.random.uniform((32, 1), dtype=tf.float32)

