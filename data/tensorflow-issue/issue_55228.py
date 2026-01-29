# tf.random.uniform((B, 50), dtype=tf.float32) ← Input shape: (batch_size, 50) feature vectors

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer with sigmoid activation, as per the example
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        return self.dense(inputs)

def my_model_function():
    # Instantiate MyModel without additional initialization/weights, 
    # as typical weights are randomly initialized.
    return MyModel()

def GetInput():
    # Return a batch of inputs matching (batch_size, 50) shape.
    # Using batch size 10 arbitrarily.
    return tf.random.uniform((10, 50), dtype=tf.float32)

