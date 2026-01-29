# tf.random.uniform((B, T), dtype=tf.int32) ← input is a batch of sequences of integer token IDs (e.g., words), shape: [batch, timesteps]

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model replicates a basic Keras LSTM stack as described in the issue:
    embedding -> LSTM layer.
    
    Assumptions and notes:
    - Input shape is (batch_size, timesteps), each element is an integer token ID for embedding.
    - Output is the last hidden state of the LSTM layer (default Keras behavior).
    - The issue reported concerned "channels_first" data format interfering with LSTM shape assumptions.
      Here, we explicitly assume channels_last or default RNN usage, so no special data_format setting.
    """
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
        self.lstm = tf.keras.layers.LSTM(128)
        
    def call(self, inputs, training=False):
        """
        Forward pass:
        inputs: int tensor shape (batch, timesteps)
        returns: tensor shape (batch, 128) - final LSTM output
        """
        x = self.embedding(inputs)
        x = self.lstm(x)
        return x

def my_model_function():
    # Return an instance of MyModel with standard initialization.
    return MyModel()

def GetInput():
    # Generate a random input tensor of shape (batch_size=32, timesteps=10)
    # Each integer token is in range [0, 999] (embedding input_dim)
    batch_size = 32
    timesteps = 10
    inputs = tf.random.uniform(shape=(batch_size, timesteps), minval=0, maxval=1000, dtype=tf.int32)
    return inputs

