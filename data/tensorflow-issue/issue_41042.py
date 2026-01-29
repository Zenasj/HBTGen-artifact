# tf.random.uniform((2, 3, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(
            units=8,
            return_sequences=True,
            return_state=False)
        self.dense = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs, training=False):
        # inputs is a tuple: (x, mask)
        x, mask = inputs
        # Pass inputs through LSTM with mask, then dense
        lstm_out = self.lstm(x, mask=mask, training=training)
        out = self.dense(lstm_out)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of (x, mask) matching MyModel's expected input
    
    batch_size = 2
    sequence_length = 3
    feature_dim = 1  # inferred from the issue input shape
    
    # Input tensor with shape (batch_size, sequence_length, feature_dim)
    x = tf.random.uniform((batch_size, sequence_length, feature_dim), dtype=tf.float32)
    
    # Create a mask tensor: boolean mask of shape (batch_size, sequence_length)
    # Use tf.sequence_mask with random sequence lengths per batch element
    seq_lens = tf.random.uniform(
        shape=(batch_size,), minval=0, maxval=sequence_length + 1, dtype=tf.int32)
    
    mask = tf.sequence_mask(seq_lens, maxlen=sequence_length)
    
    # Match the reversal in the original code [..., ::-1]?
    # The original code reversed mask sequence dimension; it's not semantically necessary, but to be faithful:
    mask = mask[:, ::-1]
    
    return (x, mask)

