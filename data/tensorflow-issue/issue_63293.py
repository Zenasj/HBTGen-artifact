# tf.random.uniform((B, window_size, 1), dtype=tf.float32) ← inferred input shape for LSTM input (batch, timesteps, features)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, window_size=10, n_steps_out=5):
        super().__init__()
        # Inferred from original code:
        # Three stacked LSTM layers with Dropout in between, ending in Dense output
        self.lstm1 = tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(window_size, 1))
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(32, return_sequences=True)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.lstm3 = tf.keras.layers.LSTM(16, return_sequences=False)
        self.dense = tf.keras.layers.Dense(n_steps_out)
    
    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.lstm3(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Provide default window_size and n_steps_out consistent with assumed defaults
    # (These values are arbitrary defaults—adjust as needed)
    window_size = 10
    n_steps_out = 5
    model = MyModel(window_size=window_size, n_steps_out=n_steps_out)
    # Build the model by calling it on a dummy input, so weights are initialized
    dummy_input = tf.random.uniform((1, window_size, 1), dtype=tf.float32)
    model(dummy_input)
    return model

def GetInput():
    # Return a random tensor input of shape (batch_size, window_size, 1)
    # batch_size and window_size must match model input
    batch_size = 4
    window_size = 10  # must match MyModel default or passed parameter
    return tf.random.uniform((batch_size, window_size, 1), dtype=tf.float32)

