# tf.random.uniform((B, None, 100), dtype=tf.float32), lengths tensor shape (B,), dtype=tf.int64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the provided example: LSTM with 100 cells, masking enabled by sequence mask, followed by Dense(1, sigmoid)
        self.mask_layer = tf.keras.layers.Lambda(lambda x: tf.sequence_mask(x), name="sequence_mask_lambda")
        self.lstm = tf.keras.layers.LSTM(100)
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        # inputs is a tuple: (sequence_tensor, lengths_tensor)
        x, lengths = inputs
        mask = self.mask_layer(lengths)
        x = self.lstm(x, mask=mask, training=training)
        output = self.classifier(x)
        return output

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput(batch_size=32):
    # Returns a tuple (input_sequences, lengths) matching model input expectations.
    # input_sequences shape: (batch_size, variable_length, 100) -- variable length is represented as None.
    # Since tensor shape must be defined for tf.random.uniform, we pick a variable length, e.g. 50.
    # lengths shape: (batch_size,), values <= sequence length (50)
    seq_len = 50
    input_tensor = tf.random.uniform((batch_size, seq_len, 100), dtype=tf.float32)
    # Random lengths between 1 and seq_len
    lengths_tensor = tf.random.uniform((batch_size,), minval=1, maxval=seq_len+1, dtype=tf.int64)
    return (input_tensor, lengths_tensor)

