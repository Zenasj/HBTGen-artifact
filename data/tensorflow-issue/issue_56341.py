# tf.random.uniform((B, 2), dtype=tf.float32) ← Input is a batch of 2D vectors representing points (x1, x2)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the same architecture as the original Sequential model,
        # but constructed as a tf.keras.Model subclass.
        self.dense1 = tf.keras.layers.Dense(100, activation='elu')
        self.dense2 = tf.keras.layers.Dense(100, activation='elu')
        self.dense3 = tf.keras.layers.Dense(50, activation='elu')
        self.dense4 = tf.keras.layers.Dense(50, activation='elu')
        self.dense5 = tf.keras.layers.Dense(10, activation='elu')
        self.dense6 = tf.keras.layers.Dense(10, activation='elu')
        self.out_layer = tf.keras.layers.Dense(1, activation='linear')
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense6(x)
        return self.out_layer(x)

def my_model_function():
    # Create an instance of MyModel and compile it with same params as original
    model = MyModel()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def GetInput():
    # Return a batch of input points shaped (batch_size, 2) matching the original x_in shape.
    # Assume batch size 100 for example; values sampled from the ranges of x1 and x2 in original code.
    # x1 in [0, 4*pi], x2 in [2, 20]
    batch_size = 100
    x1 = tf.random.uniform(shape=(batch_size, 1), minval=0.0, maxval=4*np.pi, dtype=tf.float32)
    x2 = tf.random.uniform(shape=(batch_size, 1), minval=2.0, maxval=20.0, dtype=tf.float32)
    x_input = tf.concat([x1, x2], axis=1)  # shape (batch_size, 2)
    return x_input

