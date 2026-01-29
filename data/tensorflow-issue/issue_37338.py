# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape is (batch_size, 5) as used in the original example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential model as originally shown causing the issue in TF 2.1.0,
        # composed here explicitly into layers for clarity and to avoid the save/load issue.
        self.dense1 = layers.Dense(2)
        self.dense2 = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel 
    return MyModel()

def GetInput():
    # Return a random input tensor consistent with the model input shape (batch_size, 5)
    # batch_size is set to a default 4 for testing.
    return tf.random.uniform((4, 5), dtype=tf.float32)

