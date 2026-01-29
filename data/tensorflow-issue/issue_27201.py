# tf.random.uniform((B, T), dtype=tf.int32) ← Input shape is a batch of integer sequences (token IDs)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer for vocabulary size 0x1000 (4096), output dim 128
        self.embedding = tf.keras.layers.Embedding(input_dim=0x1000, output_dim=128)
        # LSTM layer with 512 units
        self.lstm = tf.keras.layers.LSTM(512)
        # Dense layer with sigmoid activation for binary classification
        self.dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        """
        Forward pass for the model.
        Args:
          inputs: A Tensor of shape (batch_size, sequence_length), dtype int32 or int64.
        Returns:
          Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random integer tensor as input of shape (batch_size=4, sequence_length=10)
    # Token IDs range from 0 to 0x0FFF (4095) which matches Embedding input_dim
    batch_size = 4
    sequence_length = 10
    vocab_size = 0x1000  # 4096
    return tf.random.uniform((batch_size, sequence_length), minval=0, maxval=vocab_size, dtype=tf.int32)

