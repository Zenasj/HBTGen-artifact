# tf.random.uniform((N, T, n), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n, mask_value, *args, **kwargs):
        super().__init__(name='MyModel', *args, **kwargs)
        self.mask_value = mask_value
        self.n = n
        # Using LSTM with linear activation for demonstration as in the example
        self.lstm = tf.keras.layers.LSTM(self.n, return_sequences=True, activation='linear')

    def call(self, inputs, training=None, mask=None):
        # Create mask: True where inputs differ from mask_value (i.e. not padding)
        # Shape: (batch_size, seq_len)
        mask = tf.cast(tf.reduce_sum(inputs - self.mask_value, axis=-1) != 0, dtype=tf.bool)
        # Pass inputs through LSTM with masking applied
        x = self.lstm(inputs, mask=mask)
        return x

def my_model_function():
    # Instantiate the model with parameters matching those in the example
    n = 2          # feature dimension for input and LSTM units
    mask_value = -1.0
    return MyModel(n, mask_value)

def GetInput():
    # Generate a random input tensor simulating padded sequences with mask_value at padded positions
    N = 32   # batch size
    T = 10   # sequence length (max length after padding)
    n = 2    # feature dimension

    mask_value = -1.0

    import numpy as np
    np.random.seed(1)

    # Initialize array with mask_value representing padded inputs
    X = np.ones((N, T, n), dtype=np.float32) * mask_value

    # For each sequence, randomly choose a length l < T and fill the first l steps with random data
    for i in range(N):
        l = np.random.randint(1, T)
        values = np.random.random(size=(l, n)).astype(np.float32)
        X[i, :l] = values

    return tf.convert_to_tensor(X, dtype=tf.float32)

