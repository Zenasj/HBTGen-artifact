# tf.random.uniform((1, None, 32), dtype=tf.float32) 
# Input shape matches: batch_size=1, time_steps=None (variable length), features=32

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A workaround "stateful" LSTM model by managing LSTM states explicitly as model inputs and outputs.
    This avoids using stateful=True in the LSTM layer, which cannot be converted to TFLite due to resource variable issues.

    The model mimics a stateful LSTM by accepting hidden and cell states as inputs and outputing new states.
    It processes a single timestep at a time (time_steps=1) in this example.
    """

    def __init__(self, lstm_units=256, feature_dim=32, output_dim=256):
        super().__init__()
        self.lstm_units = lstm_units
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # We assume input shape: (batch=1, time_steps=1, feature_dim)
        # LSTM unrolls one timestep at a time here to mimic statefulness externally.
        
        # LSTM with return_state=True but stateful=False
        self.lstm_cell = tf.keras.layers.LSTM(
            units=self.lstm_units,
            return_sequences=True,
            return_state=True,
            stateful=False
        )
        
        # Optional Dense output to verify output shape if needed (commented out)
        # self.dense = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs):
        """
        Forward method.

        Args:
            inputs: tuple of 3 tensors:
              - x: shape (1, time_steps, feature_dim), e.g. (1, 1, 32)
              - prev_h: previous hidden state, shape (1, lstm_units)
              - prev_c: previous cell state, shape (1, lstm_units)
        
        Returns:
            output: sequence output, shape (1, time_steps, lstm_units)
            new_h: new hidden state, shape (1, lstm_units)
            new_c: new cell state, shape (1, lstm_units)
        """
        x, prev_h, prev_c = inputs
        
        # Carry out LSTM call with initial states
        lstm_output, new_h, new_c = self.lstm_cell(x, initial_state=[prev_h, prev_c])
        
        # If you want to map outputs through a Dense layer, uncomment below
        # output = self.dense(lstm_output)
        
        return lstm_output, new_h, new_c

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a tuple of random tensors matching MyModel input signatures:
      - x: random float32 tensor shape (1, 1, 32) : a single timestep input batch for batch_size=1
      - prev_h: random float32 tensor shape (1, 256) : initial hidden state
      - prev_c: random float32 tensor shape (1, 256) : initial cell state
    """
    batch_size = 1
    time_steps = 1   # must process one timestep at a time for this stateful workaround
    feature_dim = 32
    lstm_units = 256
    
    x = tf.random.uniform((batch_size, time_steps, feature_dim), dtype=tf.float32)
    prev_h = tf.zeros((batch_size, lstm_units), dtype=tf.float32)
    prev_c = tf.zeros((batch_size, lstm_units), dtype=tf.float32)
    return (x, prev_h, prev_c)

