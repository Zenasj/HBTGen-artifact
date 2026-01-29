# tf.random.normal((nb_samples, 2), dtype=tf.float32)  ← inferred input shape from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple Dense layer matching the example in the issue
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs):
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Based on the example: nb_samples=10, input shape = (10, 2)
    nb_samples = 10
    input_shape = (nb_samples, 2)
    return tf.random.normal(input_shape, dtype=tf.float32)

