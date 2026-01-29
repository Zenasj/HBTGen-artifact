# tf.random.uniform((B, 1, 52), dtype=tf.float32)  # Inputs shape: batch, time=1, features=52

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm_units = 200
        self.time_steps = 1  # As per original input shape (1, 52)
        self.input_features = 52
        self.output_dim = 13

        # Since tf.lite.experimental.nn.TFLiteLSTMCell and dynamic_rnn are TF1.x 
        # APIs unavailable natively in TF2.x, we approximate using standard tf.keras LSTM cell 
        # under assumption this example is run with appropriate 1.15 compatible environment.

        # However, to meet the task requirements (TF 2.20+ compatible code),
        # we reconstruct the model with tf.keras.layers.LSTMCell and tf.keras.layers.RNN,
        # and simulate the initial_state handling similarly.

        # Create 1-layer LSTM cell wrapped in RNN with return_sequences=True, return_state=True
        self.lstm_cell = tf.keras.layers.LSTMCell(self.lstm_units, name="rnn0")
        # Use tf.keras.layers.RNN wrapper with time_major=True equivalent (use input as batch major and time steps=1)
        # Since dynamic_rnn expects time_major inputs, and TF2 does not have tf.lite.experimental.nn.dynamic_rnn,
        # we simulate similar behavior with tf.keras.layers.RNN.

        self.rnn_layer = tf.keras.layers.RNN(
            self.lstm_cell,
            return_sequences=True,
            return_state=True,
            time_major=False,  # inputs shape: batch, time, features => time_major=False
            name="stacked_rnn"  # single layer stacked_rnn with one cell
        )
        self.dense = tf.keras.layers.Dense(self.output_dim)

    def call(self, inputs):
        """
        Args:
          inputs: tuple of three tensors
            - inputs[0]: Input sequence tensor shape (batch, time=1, features=52)
            - inputs[1]: initial hidden state h, shape (batch, 200)
            - inputs[2]: initial cell state c, shape (batch, 200)
        
        Returns:
          output: Tensor with shape (batch, time=1, 13)
          h_out: final hidden state, (batch, 200)
          c_out: final cell state, (batch, 200)
        """
        x, initial_h, initial_c = inputs

        # RNN initial state is list of [h, c]
        initial_state = [initial_h, initial_c]

        # Pass through the RNN layer with initial state
        # Output shape: sequences: (batch, time=1, units=200)
        # states: h and c (batch, units)
        rnn_output, h_out, c_out = self.rnn_layer(x, initial_state=initial_state)

        # Apply dense to each time step (time=1)
        # rnn_output shape: (batch, 1, 200) -> dense applies on last dim
        output = self.dense(rnn_output)  # (batch, 1, 13)

        return output, h_out, c_out


def my_model_function():
    return MyModel()


def GetInput():
    """
    Returns:
        Tuple of inputs matching MyModel's call:
        (inputs_seq, initial_h, initial_c)
        
        - inputs_seq: random tensor with shape (batch, 1, 52), dtype float32
        - initial_h: random tensor with shape (batch, 200), dtype float32
        - initial_c: random tensor with shape (batch, 200), dtype float32
        
    Assumption:
    - batch size is arbitrarily chosen as 4
    - input features 52, sequence length 1 as per original model
    """
    batch_size = 4
    input_seq = tf.random.uniform((batch_size, 1, 52), dtype=tf.float32)
    initial_h = tf.random.uniform((batch_size, 200), dtype=tf.float32)
    initial_c = tf.random.uniform((batch_size, 200), dtype=tf.float32)
    return (input_seq, initial_h, initial_c)

