# tf.random.uniform((B, T, F), dtype=tf.float32) ← typical RNN input shape (batch, timesteps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, go_backwards=False, units=32, **kwargs):
        super().__init__(**kwargs)
        # Store go_backwards flag to support Bidirectional wrapper
        self.go_backwards = go_backwards
        # Underlying LSTM layer with matching go_backwards parameter
        self.lstm = tf.keras.layers.LSTM(units, return_sequences=True, go_backwards=self.go_backwards)

    def call(self, inputs):
        return self.lstm(inputs)

    def get_config(self):
        # Include go_backwards to support serialization in Bidirectional wrapper
        config = super().get_config()
        config.update({'go_backwards': self.go_backwards})
        return config

def my_model_function():
    # Return a MyModel instance with default parameters.
    # The go_backwards param is left False by default.
    return MyModel()

def GetInput():
    # Return a random tensor input typical for RNNs: (batch, timesteps, features)
    # Using a batch size of 4, 10 timesteps, and 8 features (arbitrary choice)
    return tf.random.uniform((4, 10, 8), dtype=tf.float32)

