# tf.random.uniform((B, 2), dtype=tf.float32)  # Input shape inferred from X_train which has 2 features per sample

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model architecture reflecting the user code, a Sequential model with 3 Dense layers using sigmoid activations
        self.dense1 = tf.keras.layers.Dense(units=32, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(units=16, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(units=1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with the model's input
    # Assuming batch size of 8 for demonstration, with 2 features each (matching the original dataset)
    # Use float32 dtype as typical for tensorflow inputs
    batch_size = 8
    feature_dim = 2
    # Generate uniform random input in range similar to normalized data ~[-1,1]
    return tf.random.uniform((batch_size, feature_dim), minval=-1, maxval=1, dtype=tf.float32)

