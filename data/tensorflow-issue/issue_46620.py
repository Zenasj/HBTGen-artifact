# tf.random.uniform((1, 925, 3), dtype=tf.float32) ← Inferred input shape and dtype from the original Keras model input: batch_size=1, sequence_length=925, features=3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the original model described in the issue:
        # Sequential model with input shape (1, 925, 3)
        # Layers: Dense(8) -> 4 stacked LSTMs (64 units each, return_sequences=True) -> softmax activation
        self.dense = tf.keras.layers.Dense(8)
        self.lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(64, return_sequences=True)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        x = self.lstm3(x, training=training)
        x = self.lstm4(x, training=training)
        output = self.softmax(x)
        return output

def my_model_function():
    # Return an instance of MyModel, with default initialized weights
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape (batch=1, seq_len=925, features=3)
    # Use float32 as default dtype for Keras models
    return tf.random.uniform((1, 925, 3), dtype=tf.float32)

