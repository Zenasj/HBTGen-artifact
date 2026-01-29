# tf.random.uniform((4, 2), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple XOR model like the example in the issue
        self.hidden = tf.keras.layers.Dense(2, activation="sigmoid")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x = self.hidden(inputs)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile with Adam optimizer and MSE loss as per the example
    model.compile(loss="mse", optimizer="adam")
    return model

def GetInput():
    # The model expects shape (4, 2) float input based on XOR dataset:
    # 4 samples, 2 features each
    # Return a float32 tensor of shape (4, 2)
    # Using random uniform for variability, but batch size fixed at 4 as in example
    return tf.random.uniform((4, 2), minval=0, maxval=1, dtype=tf.float32)

