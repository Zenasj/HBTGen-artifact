# tf.random.uniform((1, 64), dtype=tf.int32) ← inferred from input_spec: shape [1,64], dtype int32

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A single Dense layer with 10 units as per the original example
        self.dense1 = tf.keras.layers.Dense(10)
        
    def call(self, inputs):
        # Forward pass through the dense layer
        return self.dense1(inputs)

def my_model_function():
    # Return an instance of MyModel - no special initialization needed here
    return MyModel()

def GetInput():
    # Return a random tensor that matches the input spec used in the issue:
    # batch size 1, input dimension 64, dtype int32 as per original example
    return tf.random.uniform((1, 64), minval=0, maxval=100, dtype=tf.int32)

