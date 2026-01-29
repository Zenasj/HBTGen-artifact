# tf.random.uniform((B, T, F), dtype=tf.float32)  # Input shape: batch_size x time_steps x num_features

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A model demonstrating usage of a GridLSTMCell-like RNN layer.
    Since tf.contrib.rnn.GridLSTMCell is deprecated and not available in TF 2.x, this model uses
    standard LSTMCell layers wrapped in a custom RNN structure to mimic the example usage pattern.

    Notes:
    - The original GridLSTMCell concept involves multidimensional LSTM states combining vertical
      and horizontal recurrences. TensorFlow's contrib code and documentation are sparse.
    - This model fuses forward and backward "grid" LSTMs as submodules and outputs the concatenated result,
      loosely inspired by bidirectional grid RNN.
    - Input shape: (batch_size, time_steps, num_features)
    - Output shape: (batch_size, time_steps, 2 * num_units)
    """
    def __init__(self, num_units=8, **kwargs):
        super().__init__(**kwargs)
        self.num_units = num_units
        
        # Forward LSTM cell simulating grid RNN cell with num_units units.
        self.cell_fw = tf.keras.layers.LSTMCell(num_units)
        # Backward LSTM cell simulating grid RNN cell with num_units units.
        self.cell_bw = tf.keras.layers.LSTMCell(num_units)
        
        # Use keras RNN wrappers to unroll cells statically along time axis.
        self.rnn_fw = tf.keras.layers.RNN(self.cell_fw, return_sequences=True, go_backwards=False)
        self.rnn_bw = tf.keras.layers.RNN(self.cell_bw, return_sequences=True, go_backwards=True)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, time_steps, num_features)
        
        # Forward pass
        output_fw = self.rnn_fw(inputs, training=training)  # shape: (B, T, num_units)
        # Backward pass
        output_bw = self.rnn_bw(inputs, training=training)  # shape: (B, T, num_units)
        # reverse backward output to align time dimension forward
        output_bw = tf.reverse(output_bw, axis=[1])
        
        # Concatenate forward and backward outputs along feature axis
        output = tf.concat([output_fw, output_bw], axis=-1)  # shape: (B, T, 2 * num_units)
        return output

def my_model_function():
    # Return an instance of MyModel with default units (8)
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the expected shape of MyModel.
    # Assumptions:
    # - batch_size = 1 for simplicity
    # - time_steps = 1 or more (use 5 for richer example)
    # - num_features = 1 as per example in original issue
    batch_size = 1
    time_steps = 5
    num_features = 1
    return tf.random.uniform(shape=(batch_size, time_steps, num_features), dtype=tf.float32)

