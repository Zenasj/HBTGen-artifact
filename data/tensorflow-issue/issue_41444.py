# tf.random.uniform((batch_size, None, C), dtype=tf.float32) ← input is variable length sequences with features C=5, batch_size inferred as 128 or 750 in example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define three stacked LSTM layers similar to the example from the issue
        # First LSTM has batch_input_shape=(batch_size, T, C) but here we will keep it flexible using dynamic batch and timestep dims,
        # matching input shape of (batch_size, None, 5) for variable sequence length and 5 features
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(16, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(8)
        self.output_layer = tf.keras.layers.Dense(1)
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs, training=training)
        x = self.lstm2(x, training=training)
        x = self.lstm3(x, training=training)
        output = self.output_layer(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # According to the example snippet in the issue:
    # Input shape: (batch_size, timesteps, C)
    # Using batch_size=128 as a stable example,
    # features C=5 (from the code in issue)
    # timesteps variable length; let's use 100 for a reasonable sequence length here
    
    batch_size = 128
    timesteps = 100  # Reasonable arbitrary seq length for testing
    C = 5

    # Use tf.random.uniform with float32 dtype to generate test input
    # This matches the LSTM input shape requirements (batch_size, timesteps, features)
    input_tensor = tf.random.uniform((batch_size, timesteps, C), dtype=tf.float32)
    return input_tensor

