# tf.random.normal(shape=(100, 1, 10)) for features input (batch=100, steps=1, features=10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple Dense layer matching the example in the issue
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid', name="L0")

    def call(self, inputs, training=False):
        # inputs assumed shape (batch, 1, 10), we reduce the middle dimension or reshape to (batch, 10)
        # as the original input layer has shape=[10] (without time dimension)
        # since the example input is (100, 1, 10), flatten or squeeze dimension 1
        # This matches the Input(shape=[10]) used in the reported example
        
        # Remove the middle dimension: (batch, 1, 10) -> (batch, 10)
        if len(inputs.shape) == 3 and inputs.shape[1] == 1:
            x = tf.squeeze(inputs, axis=1)
        else:
            x = inputs
        
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on the example, (batch=100, seq=1, features=10), dtype float32
    return tf.random.normal(shape=(100, 1, 10), dtype=tf.float32)

