# tf.random.uniform((B, 1, 1), dtype=tf.float32) for each input timeseries (3 timeseries)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.window_size = 1  # from example
        
        # Create 3 LSTM branches, one per input timeseries
        self.lstm_layers = [tf.keras.layers.LSTM(32) for _ in range(3)]
        self.dense_per_ts = [tf.keras.layers.Dense(units=1) for _ in range(3)]
        
        # After concatenation, flatten and a final dense layer
        self.flatten = tf.keras.layers.Flatten()
        self.final_dense = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=False):
        """
        inputs: list or tuple of 3 tensors, each shape (batch_size, window_size=1, channels=1)
        """
        # Process each timeseries input separately through lstm + dense
        processed = []
        for i, inp in enumerate(inputs):
            x = self.lstm_layers[i](inp)
            x = self.dense_per_ts[i](x)
            processed.append(x)
        # Concatenate outputs along last dimension (features)
        concat_out = tf.concat(processed, axis=-1)
        # Flatten then final dense
        flat = self.flatten(concat_out)
        out = self.final_dense(flat)
        return out


def my_model_function():
    """
    Instantiate and return model instance as per the user's example.
    """
    return MyModel()


def GetInput():
    """
    Generate a list of 3 input tensors matching MyModel input:
    Each tensor shape: (batch_size=32, window_size=1, channels=1)
    Values are random floats from uniform distribution.
    """
    batch_size = 32
    window_size = 1
    channels = 1
    input_list = [tf.random.uniform((batch_size, window_size, channels), dtype=tf.float32)
                  for _ in range(3)]
    return input_list

