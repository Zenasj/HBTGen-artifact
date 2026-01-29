# tf.random.uniform((B, 5, 1), dtype=tf.float32) ← Input shape inferred from model input_shape=(n_steps=5, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the structure from the issue:
        # LSTM layer with 200 units, relu activation, input shape (5,1)
        self.lstm = tf.keras.layers.LSTM(200, activation='relu', input_shape=(5, 1))
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(50, activation='relu')
        self.out = tf.keras.layers.Dense(1)  # output layer, 1 node
        
    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x

def my_model_function():
    # Instantiate the model (weights randomly initialized, ready to be compiled and trained)
    return MyModel()

def GetInput():
    # Generate a random input tensor that matches the input shape: (batch_size, n_steps=5, features=1)
    # Use batch size 32 as a typical batch size from training
    # Use float32 data type as per values.astype('float32')
    
    # The model expects shape (batch, 5, 1)
    import numpy as np
    batch_size = 32
    n_steps = 5
    features = 1
    # Use random normal or uniform, uniform is common—let's do uniform between 0 and 1
    input_tensor = tf.random.uniform((batch_size, n_steps, features), dtype=tf.float32)
    return input_tensor

