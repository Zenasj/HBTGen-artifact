# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Inferred input shape from Flatten input_shape=(28, 28)

import tensorflow as tf
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the original Sequential model layers explicitly because the snippet defines a keras.Sequential
        self.flatten = keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = keras.layers.Dense(100, activation="relu")
        self.dense2 = keras.layers.Dense(10, activation="sigmoid")
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel as per the original snippet
    model = MyModel()
    return model

def GetInput():
    # Return a random tensor matching the input shape expected by the model: batch dim is arbitrary, 
    # using batch size 32 here as a common default
    batch_size = 32
    input_tensor = tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)
    return input_tensor

