# tf.random.uniform((B, T, F), dtype=tf.float32)  ← Assuming input is a batch of sequences compatible with LSTM layers

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, hid_size=32, num_layers=3):
        super().__init__()
        # Instead of storing layers in a plain list, use a tf.keras.LayerList container
        # to ensure that Keras properly tracks them and their variables for gradient computation.
        self.lstm_layers = tf.keras.layers.StackedRNNCells(
            [layers.LSTMCell(hid_size) for _ in range(num_layers)]
        )
        # Wrap StackedRNNCells with RNN layer to use in call()
        self.rnn = layers.RNN(self.lstm_layers, return_sequences=False)

    def call(self, x, training=False):
        # Forward through stacked LSTM layers via the RNN wrapper
        return self.rnn(x, training=training)

def my_model_function():
    # Instantiate the model with default hidden size and number of layers
    return MyModel()

def GetInput():
    # Construct a sample input tensor matching expected input shape of (batch, time_steps, features)
    # Assuming batch size 4, sequence length 10, and input feature size 8 as a reasonable example
    return tf.random.uniform((4, 10, 8), dtype=tf.float32)

