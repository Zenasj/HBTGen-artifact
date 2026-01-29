# tf.random.uniform((B, None, 512), dtype=tf.float32) ← Input is ragged with variable time steps and feature size 512

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 32 units, expecting ragged input with variable length sequences
        self.lstm = layers.LSTM(32, return_sequences=True, dropout=0.4)
        # TimeDistributed Dense to classify each timestep into 13 classes with softmax activation
        self.time_distributed = layers.TimeDistributed(layers.Dense(13, activation='softmax'))

    def call(self, inputs, training=False):
        # The inputs are expected to be a RaggedTensor of shape [batch, None, 512]
        # LSTM layer in TF 2.5+ supports ragged input directly
        x = self.lstm(inputs, training=training)
        # Apply the TimeDistributed Dense layer on the LSTM output
        output = self.time_distributed(x)
        # The output will be a RaggedTensor if input is RaggedTensor
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a ragged input tensor matching shape [batch, None, 512]
    # For demonstration, batch size 3, with variable lengths for time dimension
    batch_size = 3
    feature_size = 512
    # Define row_splits for ragged tensor - 3 sequences with lengths 2, 5, 4
    row_lengths = [2, 5, 4]
    # Total timesteps = sum(row_lengths)
    total_timesteps = sum(row_lengths)
    # Create uniform random values for flattened tensor: [total_timesteps, feature_size]
    values = tf.random.uniform((total_timesteps, feature_size), dtype=tf.float32)
    # Build the RaggedTensor using row_lengths
    ragged_input = tf.RaggedTensor.from_row_lengths(values, row_lengths)
    # Shape: [3, None, 512]
    return ragged_input

