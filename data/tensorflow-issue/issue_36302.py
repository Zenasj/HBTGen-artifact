# tf.random.uniform((10, None, 2), dtype=tf.float32) ← Input shape: batch=10, time_steps variable, features=2

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model layers as per the example in the issue:
        # Variable-length time dimension input with 2 features
        self.lstm_size = 256
        self.lstm = tf.keras.layers.LSTM(self.lstm_size)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        # inputs shape: (batch, time_steps, features)
        # Forward pass through LSTM and Dense
        x = self.lstm(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Instantiate and return the model
    model = MyModel()
    # Normally would call build or run once to initialize weights, but here just return the raw model.
    return model

def GetInput():
    # Generate a valid input tensor that matches the model input
    # Fixed batch size 10, variable time steps (choose 8 for example), features=2 matching issue input
    batch_size = 10
    time_steps = 8
    feature_dimension = 2

    # Use tf.random.uniform with float32 which is typical for inputs
    return tf.random.uniform((batch_size, time_steps, feature_dimension), dtype=tf.float32)

