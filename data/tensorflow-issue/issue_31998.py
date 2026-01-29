# tf.random.uniform((1, 100, 100), dtype=tf.float32), tf.random.uniform((1, 100), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using GRU layer, trying to match the original example's units=100
        self.gru_layer = tf.keras.layers.GRU(100)
        # Using LSTM layer with RNN(LSTMCell) workaround to avoid persistent=True bug
        self.lstm_layer = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(100))

    @tf.function
    def call(self, inputs):
        # inputs is assumed to be a tuple: (x_gru, y_gru), (s_lstm, h_lstm, c_lstm)
        # For clarity, expect inputs = (x_gru, y_gru, s_lstm, h_lstm, c_lstm)
        # which follows from the issue examples.

        # Unpack inputs
        x_gru, y_gru, s_lstm, h_lstm, c_lstm = inputs

        # GRU call: y_gru as sequence input, x_gru as initial state
        # Match original signature: layer(y, initial_state=x)
        gru_output = self.gru_layer(y_gru, initial_state=x_gru)

        # LSTM call: use RNN(LSTMCell) with initial_state tuple (h, c)
        lstm_output = self.lstm_layer(s_lstm, initial_state=[h_lstm, c_lstm])

        # Compare the outputs by norm of difference as a single float tensor:
        # This represents a numeric difference to reflect the original issue's comparison motives.
        diff = tf.norm(gru_output - lstm_output)

        # Returning scalar difference as output (could also return a boolean by threshold if needed)
        return diff

def my_model_function():
    return MyModel()

def GetInput():
    # Construct inputs matching Model call signature:
    # From the issue:
    # x (initial_state for GRU) shape=(1, 100)
    # y (sequence input for GRU) shape=(1, 100, 100)
    # s (sequence input for LSTM) shape=(1, 100, 100)
    # h, c (initial states for LSTM) shape=(1, 100) each

    x_gru = tf.random.uniform((1, 100), dtype=tf.float32)        # initial_state for GRU
    y_gru = tf.random.uniform((1, 100, 100), dtype=tf.float32)   # sequence input for GRU
    s_lstm = tf.random.uniform((1, 100, 100), dtype=tf.float32)  # sequence input for LSTM
    h_lstm = tf.random.uniform((1, 100), dtype=tf.float32)       # initial hidden state for LSTM
    c_lstm = tf.random.uniform((1, 100), dtype=tf.float32)       # initial cell state for LSTM

    return (x_gru, y_gru, s_lstm, h_lstm, c_lstm)

