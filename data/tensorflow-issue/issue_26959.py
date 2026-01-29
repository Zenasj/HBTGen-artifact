# tf.random.uniform((B, T, F), dtype=tf.float32) ← Assuming input shape (batch_size, timesteps, features) for LSTM

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two-layer LSTM model with dropout as described in the issue
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(256)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The issue does not specify input shape explicitly.
    # Common shape for sequence models: (batch_size, timesteps, features)
    # We'll assume batch_size=4, timesteps=10, features=8 for demonstration.
    # Use tf.float32 dtype as typical for Keras inputs.
    batch_size = 4
    timesteps = 10
    features = 8
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

