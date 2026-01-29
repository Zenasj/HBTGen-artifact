# tf.random.uniform((B, sequence_length, raw_data_channels), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras

# Assumptions:
# - Input shape is (batch_size, sequence_length, raw_data_channels)
# - The issue arises from using Flatten which fails under TF 2.16+
# - The recommended workaround is to use a Reshape layer instead of Flatten
# - raw_data_channels is the last dimension of the original data
# - We'll use example dummy sizes for sequence_length and raw_data_channels for GetInput()

sequence_length = 10          # Example fixed sequence length (inferred from issue context)
raw_data_channels = 5         # Example number of features per timestep

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instead of Flatten, we use Reshape to flatten the time and feature dims explicitly
        # This matches: Reshape((sequence_length * raw_data_channels,))
        self.reshape = keras.layers.Reshape((sequence_length * raw_data_channels,))
        self.dense1 = keras.layers.Dense(16, activation="relu")
        self.dense2 = keras.layers.Dense(1)

    def call(self, inputs):
        # inputs shape: (batch, sequence_length, raw_data_channels)
        x = self.reshape(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel with frozen params from default init
    return MyModel()

def GetInput():
    # Provide a random tensor input with shape matching model input:
    # (batch_size, sequence_length, raw_data_channels)
    batch_size = 4  # example batch size
    input_tensor = tf.random.uniform(
        (batch_size, sequence_length, raw_data_channels), dtype=tf.float32)
    return input_tensor

