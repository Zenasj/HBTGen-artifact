# tf.random.uniform((B, 20), dtype=tf.float32) ← Assuming input shape (batch_size, 20) as per issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Define the Dense layer in __init__ (recommended over build for checkpointing consistency)
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        out = self.dense(inputs)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input expected by MyModel
    # Based on example: input shape is (batch_size=10, 20)
    return tf.random.uniform((10, 20), dtype=tf.float32)

