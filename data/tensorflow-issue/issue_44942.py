# tf.random.uniform((B, window, 1), dtype=tf.float32) ← Input shape inferred from LSTM input_shape=[window, 1]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, window=10, lstm_units=32, dense_units=1):
        super().__init__()
        # LSTM layer with specified units and input shape last dimension = 1
        self.lstm = tf.keras.layers.LSTM(units=lstm_units, input_shape=(window, 1))
        # Dense output layer
        self.dense = tf.keras.layers.Dense(units=dense_units)

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel with default window size = 10 (arbitrary assumption)
    # Users should adjust window based on their data
    return MyModel(window=10)

def GetInput():
    # Generate a random input tensor with shape (batch_size, window, 1)
    # Assuming batch size = 8, window = 10 (matches model default)
    batch_size = 8
    window = 10
    # Random uniform float32 tensor
    return tf.random.uniform((batch_size, window, 1), dtype=tf.float32)

