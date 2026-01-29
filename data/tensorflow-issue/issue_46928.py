# tf.random.uniform((batch_size, sequence_length), dtype=tf.int32) ← Input shape is (batch_size, sequence_length) with integer tokens for Embedding input

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Typical simple LSTM model from the issue:
        # Embedding input_dim=1000, output_dim=64
        # LSTM with 128 units
        # Dense output layer with 10 units (e.g. classification or regression)
        self.embedding = layers.Embedding(input_dim=1000, output_dim=64)
        self.lstm = layers.LSTM(128)
        self.dense = layers.Dense(10)

    def call(self, inputs):
        x = self.embedding(inputs)   # input shape: (batch, seq_len)
        x = self.lstm(x)             # output shape: (batch, 128)
        x = self.dense(x)            # output shape: (batch, 10)
        return x

def my_model_function():
    # Return an instance of the simple LSTM model as described
    return MyModel()

def GetInput():
    # Generate a random batch of integer sequences representing token indices:
    # Assumptions:
    # - batch_size = 4 (arbitrary small batch)
    # - sequence_length = 20 (typical short sequence)
    # - values from 0 to 999 (matching embedding input_dim)
    batch_size = 4
    sequence_length = 20
    return tf.random.uniform(
        shape=(batch_size, sequence_length), 
        minval=0, maxval=1000, 
        dtype=tf.int32
    )

