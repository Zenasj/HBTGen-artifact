# tf.random.uniform((32, 1, 100), dtype=tf.float32) ← This matches batch_size=32, sequence_len=1, embedding_size=100

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 200 units, not returning sequences, input shape fixed by batch_size and time steps
        self.lstm_layer = tf.keras.layers.LSTM(200, return_sequences=False)
        # Dense layer to project LSTM output back to embedding size (100)
        self.dense_layer = tf.keras.layers.Dense(100)

    def call(self, inputs, training=False):
        x = self.lstm_layer(inputs, training=training)
        return self.dense_layer(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Returns random input tensor matching input shape: batch_size=32, sequence_len=1, embedding_size=100
    # Use float32 dtype to be compatible by default
    batch_size = 32
    sequence_len = 1
    embedding_size = 100
    return tf.random.uniform((batch_size, sequence_len, embedding_size), dtype=tf.float32)

