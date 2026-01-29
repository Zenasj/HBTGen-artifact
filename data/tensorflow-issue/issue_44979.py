# tf.RaggedTensor shape: (batch_size, None, 1) <- variable-length sequences with 1 feature

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM layer with 5 units, processes ragged inputs with variable time steps
        self.lstm = tf.keras.layers.LSTM(5, return_sequences=False, name='LSTM')
        # Dense output layer, linear activation
        self.dense = tf.keras.layers.Dense(1, activation='linear', name='Predictor')

    def call(self, inputs, training=False):
        # inputs: a RaggedTensor of shape (batch, var_seq_len, 1)
        # LSTM supports ragged inputs natively in TF 2.x
        x = self.lstm(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Returns an instance of MyModel with standard initialization
    return MyModel()

def GetInput():
    # Returns a random RaggedTensor input matching the expected input:
    # (batch_size=4, variable sequence length up to max_seq_length=5, feature_dim=1)
    batch_size = 4
    max_seq_length = 5
    
    # For reproducibility, we pick random sequence lengths between 1 and max_seq_length inclusive
    # Use int32 values in range 0-99 as dummy data, single feature dimension
    # RaggedTensor created from flattened values with row_lengths
    
    # Random sequence lengths for each batch sample
    seq_lengths = tf.random.uniform(shape=(batch_size,), minval=1, maxval=max_seq_length+1, dtype=tf.int32)

    # Total values across batch = sum of sequence lengths
    total_values = tf.reduce_sum(seq_lengths)

    # Random int values between 0 and 99
    values = tf.random.uniform(shape=(total_values,), minval=0, maxval=100, dtype=tf.int32)

    # Construct RaggedTensor via from_row_lengths: shape [batch_size, (var_seq_len)]
    ragged = tf.RaggedTensor.from_row_lengths(values=tf.cast(values, tf.float32), row_lengths=seq_lengths)
    # Expand dims to add feature dim (last dim = 1)
    ragged_expanded = tf.expand_dims(ragged, axis=-1)
    
    return ragged_expanded

