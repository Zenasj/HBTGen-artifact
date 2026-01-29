# tf.random.uniform((512, 5000), dtype=tf.float32) ← inferred input shape from example usage

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple model with one Dense layer as in the provided test snippet
        self.dense = tf.keras.layers.Dense(4096, activation='relu', name="dense_relu")

    def call(self, inputs):
        # Call the dense layer on inputs
        # This matches the minimal model structure in example test code for TRTEngine conversion
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel as required
    return MyModel()

def GetInput():
    # Return a random tensor that matches the input expected by MyModel
    # From the example test case, input is shape (512, 5000) of float32
    return tf.random.uniform((512, 5000), dtype=tf.float32)

