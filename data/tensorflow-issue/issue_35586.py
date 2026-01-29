# tf.random.uniform((1, 2), dtype=tf.float32) ← Input shape inferred from environment state size for MountainCar-v0

import tensorflow as tf
import numpy as np
import random
from collections import deque

class MyModel(tf.keras.Model):
    def __init__(self, state_size=2, action_size=3):
        super().__init__()
        # Defining the same architecture as in the original keras.Sequential model:
        # Dense(state_size, relu), Dense(24, relu), Dense(48, relu), Dense(action_size, linear)
        self.dense1 = tf.keras.layers.Dense(state_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(48, activation='relu')
        self.out = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

def my_model_function():
    # Return an instance of MyModel with default MountainCar-v0 state and action sizes
    return MyModel()

def GetInput():
    # Generate a random tensor input matching MountainCar-v0 state shape (1, 2)
    # Single batch, 2 features, float32
    return tf.random.uniform(shape=(1, 2), dtype=tf.float32)

