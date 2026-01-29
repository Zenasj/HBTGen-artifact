# tf.random.uniform((B, T, D), dtype=tf.float32) ← Assuming input shape matches (timesteps, dim)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, timesteps=10, dim=5):
        super().__init__()
        # Using 6 stacked LSTM layers all returning sequences as per original code
        self.lstm_layers = []
        for i in range(6):
            # First LSTM layer specifies input shape, others infer automatically
            if i == 0:
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(120, activation="tanh", return_sequences=True, input_shape=(timesteps, dim)))
            else:
                self.lstm_layers.append(
                    tf.keras.layers.LSTM(120, activation="tanh", return_sequences=True))
        # Final Dense with softmax activation as per suggestion, for multi-class probabilities on last dim
        self.dense = tf.keras.layers.Dense(dim, activation="softmax")

    def call(self, inputs):
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Default timesteps and dim values can be adjusted to match your dataset
    return MyModel(timesteps=10, dim=5)

def GetInput():
    # Create a random tensor input of shape (batch_size, timesteps, dim)
    # Batch size chosen as 10 (matching the batch_size in original training)
    batch_size = 10
    timesteps = 10  # inferred default timesteps (must be consistent with MyModel)
    dim = 5         # inferred default feature dimension (must be consistent with MyModel)
    return tf.random.uniform((batch_size, timesteps, dim), minval=0, maxval=1, dtype=tf.float32)

