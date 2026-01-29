# tf.random.uniform((B, time_window_size, 1), dtype=tf.float32) ← Assuming input shape based on example: (batch, time_window_size, channels=1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, time_window_size=100):
        super().__init__()
        # Following the example sequence from the issue:
        # Conv1D with 256 filters, kernel_size=5, padding='same', relu activation
        self.conv1d = tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=5,
            padding='same',
            activation='relu',
            input_shape=(time_window_size, 1)
        )
        # MaxPooling1D with pool_size=4
        self.maxpool = tf.keras.layers.MaxPooling1D(pool_size=4)
        # LSTM with 64 units
        self.lstm = tf.keras.layers.LSTM(64)
        # Dense layer with units = time_window_size, linear activation
        self.dense = tf.keras.layers.Dense(time_window_size, activation='linear')

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.maxpool(x)
        x = self.lstm(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Defaulting to time_window_size=100 for illustration
    return MyModel(time_window_size=100)

def GetInput():
    # Create a random input tensor consistent with the assumed input shape
    # Shape: (batch_size=1, time_window_size=100, channels=1)
    time_window_size = 100
    batch_size = 1
    input_tensor = tf.random.uniform(
        shape=(batch_size, time_window_size, 1),
        dtype=tf.float32
    )
    return input_tensor

