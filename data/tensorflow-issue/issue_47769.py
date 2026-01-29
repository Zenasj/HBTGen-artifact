# tf.random.uniform((128, 256), dtype=tf.int32) ← inferred input shape from batch_size=128 and sequence_length=256

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer: input_dim=100 (vocab size), output_dim=100 (embedding size)
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=100)
        # LSTM layer with 512 units, return sequences for sequence prediction
        self.lstm = tf.keras.layers.LSTM(512, return_sequences=True)
        # TimeDistributed Dense to project LSTM output to vocab logits
        self.time_dist = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100))
    
    def call(self, x, training=False):
        """
        Forward pass:
        x - int32 tensor of shape (batch_size, sequence_length)
        """
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = self.lstm(x)       # (batch, seq_len, 512)
        x = self.time_dist(x)  # (batch, seq_len, 100)
        return x

def my_model_function():
    # Returns a compiled instance of MyModel with configured loss and optimizer
    model = MyModel()
    # Compile with SparseCategoricalCrossentropy from logits and RMSprop optimizer
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.004)
    model.compile(optimizer=opt, loss=loss)
    return model

def GetInput():
    # Generates a random batch of integer sequences with shape (128, 256), values in [0, 100)
    batch_size = 128
    seq_length = 256
    # Return int32 input tensor matching Model expected input shape and dtype
    return tf.random.uniform(shape=(batch_size, seq_length), minval=0, maxval=100, dtype=tf.int32)

