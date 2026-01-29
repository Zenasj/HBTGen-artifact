# tf.random.normal((batch_size * num_batches, num_timesteps, num_features), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a single LSTM layer similar to the one in the issue
        # LSTM units=2, input shape=(None, 40 features), return_sequences=True
        self.lstm = tf.keras.layers.LSTM(2, return_sequences=True)

    def call(self, inputs, training=False):
        # Forward pass through the LSTM layer
        return self.lstm(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel initialized without pretrained weights
    return MyModel()

def GetInput():
    # Generate a random tensor matching the input shape used in the example:
    # batch_size=16, num_batches=4 (total batch = 64), timesteps=100, features=40
    # Using tf.random.normal to match original example input
    batch_size = 16
    num_batches = 4
    num_timesteps = 100
    num_features = 40
    input_shape = (batch_size * num_batches, num_timesteps, num_features)
    return tf.random.normal(input_shape, dtype=tf.float32)

