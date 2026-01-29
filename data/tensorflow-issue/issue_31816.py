# tf.random.uniform((B, None, 5), dtype=tf.float32) ← Variable sequence length, feature size 5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Two stacked LSTM layers are created with return_sequences=True to support variable-length sequences.
        # input_shape=(None, 5) means variable sequence length, feature dim=5.
        self._lstm_0 = tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(None, 5))
        self._lstm_1 = tf.keras.layers.LSTM(8, return_sequences=True)
        self._dense = tf.keras.layers.Dense(2)

    def call(self, inputs, training=False):
        x = self._lstm_0(inputs, training=training)
        x = self._lstm_1(x, training=training)
        x = self._dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel, ready for compilation and use.
    return MyModel()

def GetInput():
    # Generates a random input tensor with:
    # - batch size: 1000 (as used in the example)
    # - variable sequence length: randomly chosen from 10 to 100 to simulate variable length batches
    # - feature dimension: 5
    batch_size = 1000
    sequence_length = tf.random.uniform([], minval=10, maxval=100, dtype=tf.int32)
    input_tensor = tf.random.uniform((batch_size, sequence_length, 5), dtype=tf.float32)
    return input_tensor

