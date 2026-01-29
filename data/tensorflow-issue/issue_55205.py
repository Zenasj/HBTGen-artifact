# tf.random.uniform((1, 8, 4), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model is a single Dense layer with output units=16, input shape (8,4)
        # reflecting the QAT model in the issue.
        self.dense = tf.keras.layers.Dense(16)
    
    def call(self, inputs):
        # inputs shape: (batch_size=1, 8, 4)
        # same as in the issue example
        return self.dense(inputs)

def my_model_function():
    # Returns an instance of the model with a Dense layer (units=16)
    return MyModel()

def GetInput():
    # Returns a random tensor with shape (1,8,4) matching model input
    return tf.random.uniform((1, 8, 4), dtype=tf.float32)

