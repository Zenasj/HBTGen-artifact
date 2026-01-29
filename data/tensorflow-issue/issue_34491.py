# tf.random.uniform((100, 20, 1), dtype=tf.float32) ← Input shape inferred from original example: (num_seqs=100, time_steps=20, features=1)

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self, time_steps=20, lstm_size=16, mask_value=0., **kwargs):
        super().__init__(**kwargs)
        # Masking layer to ignore padded values (= mask_value)
        self.mask = tf.keras.layers.Masking(mask_value=mask_value, input_shape=(time_steps, 1))
        # LSTM with return_sequences=True for sequence labeling
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
        # Final Dense layer outputs 1 scalar per timestep
        self.dense = tf.keras.layers.Dense(1)
        # Sigmoid activation for binary classification per timestep
        self.activation = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs, training=False):
        x = self.mask(inputs)
        x = self.lstm(x, training=training)
        x = self.dense(x)
        x = self.activation(x)
        return x


def my_model_function():
    """
    Returns an instance of MyModel with default parameters matching the original reported issue:
    - input shape: (20 timesteps, 1 feature)
    - mask_value=0. (matching the pad_sequences default padding)
    - LSTM size=16
    """
    return MyModel()


def GetInput():
    """
    Returns a randomly generated input tensor matching the expected input shape of MyModel:
    - shape: (100, 20, 1)
    - dtype: tf.float32
    - values follow standard normal distribution, but with padded zeros applied at end of sequence as in original example

    To simulate padding as in original issue:
    - Variable sequence lengths between 1 and 20
    - padded with zeros at the end (mask_value=0.0)
    """

    num_seqs = 100
    time_steps = 20
    feature_dim = 1

    # Generate random normal data
    X = np.random.normal(size=(num_seqs, time_steps)).astype(np.float32)

    # Generate random sequence lengths between 1 and 20 (as in original)
    lengths = np.random.randint(low=1, high=time_steps + 1, size=num_seqs)

    # Zero out values beyond each sequence length to simulate padding
    for i, length in enumerate(lengths):
        if length < time_steps:
            X[i, length:] = 0.0  # pad by zeros to match mask_value

    # Reshape to (num_seqs, time_steps, 1)
    X = np.expand_dims(X, axis=-1)

    # Convert to tf.Tensor
    return tf.convert_to_tensor(X, dtype=tf.float32)

