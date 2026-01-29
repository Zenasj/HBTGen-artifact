# tf.random.uniform((B, T, F), dtype=tf.float32)  # Assumed input shape: batch, time steps, features

import tensorflow as tf

# Constants inferred from the snippet
OUT_STEPS = 10    # Typical prediction horizon in time series examples
num_features = 5  # Number of features per time step

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer: takes input shape (batch, time, features), outputs (batch, lstm_units)
        self.lstm = tf.keras.layers.LSTM(32, return_sequences=False)
        # Dense layer with kernel initializer fixed using callable instance (note the () after zeros)
        self.dense = tf.keras.layers.Dense(
            OUT_STEPS * num_features,
            kernel_initializer=tf.initializers.zeros()
        )
        # Reshape layer to convert flat output to (batch, out_steps, features)
        self.reshape = tf.keras.layers.Reshape([OUT_STEPS, num_features])

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dense(x)
        x = self.reshape(x)
        return x


def my_model_function():
    # Return an instance of the model with proper initialization
    return MyModel()


def GetInput():
    # Return a random tensor input with shape [batch, time, features]
    # Using batch size 32 as a typical example
    batch_size = 32
    time_steps = 20  # Number of time steps in input sequence
    input_tensor = tf.random.uniform(
        (batch_size, time_steps, num_features),
        dtype=tf.float32
    )
    return input_tensor

