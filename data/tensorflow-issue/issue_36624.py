# tf.random.uniform((3, 9, 2), dtype=tf.float32)  # inferred input shape from issue dataset example

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates a single LSTM layer configured with return_state=True,
    returning the full output list (output, hidden state, cell state).
    
    This is not a Sequential model because Sequential does not support layers that
    return multiple outputs as a list (like LSTM with return_state=True).
    Instead, we manually handle the multiple outputs here.
    """
    def __init__(self, ts=9, input_dim=2, lstm_units=3):
        super().__init__()
        # LSTM layer with return_state=True returns a list: [output, h_state, c_state]
        # Input shape: (batch, ts, input_dim)
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_state=True)
        
    def call(self, inputs, training=False):
        # The LSTM call returns multiple outputs: output, hidden state, cell state
        lstm_outputs = self.lstm(inputs, training=training)
        # lstm_outputs is a list of length 3:
        # [output_sequence, hidden_state, cell_state]
        # Here, we'll just return the entire list.
        # This replicates the Functional API output behavior.
        return lstm_outputs

def my_model_function():
    # Return an instance of MyModel with default parameters matching the example
    return MyModel(ts=9, input_dim=2, lstm_units=3)

def GetInput():
    # Return a random tensor input matching shape (batch_size=3, time_steps=9, input_dim=2)
    # Use float32 dtype to match example and TensorFlow defaults
    batch_size = 3
    ts = 9
    input_dim = 2
    return tf.random.uniform((batch_size, ts, input_dim), dtype=tf.float32)

