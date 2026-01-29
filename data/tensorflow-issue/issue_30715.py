# tf.random.uniform((B, T, 512), dtype=tf.float32) ← Input batch of sequences with variable length T and feature size 512

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Stack 12 LSTMCells inside one RNN layer as in the original issue code
        cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(512) for _ in range(12)]
        )
        self.rnn = tf.keras.layers.RNN(cell)
    
    def call(self, inputs, training=False):
        # inputs shape expected: (batch_size, time_steps, 512)
        # Runs stacked LSTMCells over input sequences
        return self.rnn(inputs, training=training)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random tensor with variable sequence length matching expected input shape
    # Based on the dataset generation in the issue, sequence length (time_steps) varies up to 80
    batch_size = 64  # typical batch size from example
    max_time_steps = 80
    feature_dim = 512

    # Using max sequence length for simplicity, since tf.RNN can handle padded sequences
    # The values are uniform floats between 0 and 1
    return tf.random.uniform((batch_size, max_time_steps, feature_dim), dtype=tf.float32)

