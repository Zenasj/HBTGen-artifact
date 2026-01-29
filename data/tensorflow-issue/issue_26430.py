# tf.random.uniform((10, 3), dtype=tf.float32) ← inferred from the example "x = tf.zeros([10, 3])" input shape for the model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # single Dense layer with 256 units as per the example
        self.dense = tf.keras.layers.Dense(256)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # No special initialization needed except normal Dense layer initialization.
    return MyModel()

def GetInput():
    # Returns a random float input tensor matching expected input [10,3]
    # dtype tf.float32 as typical for Dense inputs
    return tf.random.uniform((10, 3), dtype=tf.float32)

