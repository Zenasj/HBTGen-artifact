# tf.random.uniform((1, 2), dtype=tf.float32)  # inferred input shape from example usage in issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Dense layer with 100 units, as per original example
        self.layer = tf.keras.layers.Dense(100)

    def call(self, x):
        return self.layer(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (1,2)
    return tf.random.uniform((1, 2), dtype=tf.float32)

