# tf.random.uniform((256, 144, 130), dtype=tf.float32) ← Inferred input shape for model input: batch_size=256, num_tsteps=144, num_features=130

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.batch_size = 256
        self.num_tsteps = 144
        self.num_features = 130
        self.num_units = 88
        # Define the layers following original Sequential design
        self.lstm = tf.keras.layers.LSTM(
            self.num_units, 
            batch_input_shape=(self.batch_size, self.num_tsteps, self.num_features),
            return_sequences=True, 
            stateful=False
        )
        self.time_dist = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        self.activation = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs, training=None):
        # inputs expected shape: (batch_size, num_tsteps, num_features)
        x = self.lstm(inputs)
        x = self.time_dist(x)
        x = self.activation(x)
        return x

def my_model_function():
    # Instantiate MyModel with necessary initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor matching model input shape
    # Using tf.float32 dtype to match model input expectations
    batch_size = 256
    num_tsteps = 144
    num_features = 130
    # Uniform random tensor between 0 and 1
    input_tensor = tf.random.uniform(
        shape=(batch_size, num_tsteps, num_features), 
        dtype=tf.float32
    )
    return input_tensor

