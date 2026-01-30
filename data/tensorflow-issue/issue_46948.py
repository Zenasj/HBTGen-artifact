import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(1, 52))
state_1_h = keras.Input(shape=(200,))
state_1_c = keras.Input(shape=(200,))
x1, state_1_h_out, state_1_c_out = layers.LSTM(200, return_sequences=True, input_shape=(sequence_length, 52),
                                               return_state=True)(inputs, initial_state=[state_1_h, state_1_c])
output = layers.Dense(13)(x1)

model = keras.Model([inputs, state_1_h, state_1_c],
                    [output, state_1_h_out, state_1_c_out])

x1, state_1_h_out, state_1_c_out = layers.LSTM(200, return_sequences=True, input_shape=(sequence_length, 52),
                                               return_state=True)(inputs, initial_state=[state_1_h, state_1_c])

def buildLstmLayer(inputs, num_layers, num_units):
  """Build the lstm layer.

  Args:
    inputs: The input data.
    num_layers: How many LSTM layers do we want.
    num_units: The unmber of hidden units in the LSTM cell.
  """
  lstm_cells = []
  for i in range(num_layers):
    lstm_cells.append(
        tf.lite.experimental.nn.TFLiteLSTMCell(
            num_units, forget_bias=0, name='rnn{}'.format(i)))
  lstm_layers = tf.keras.layers.StackedRNNCells(lstm_cells)
  # Assume the input is sized as [batch, time, input_size], then we're going
  # to transpose to be time-majored.
  transposed_inputs = tf.transpose(
      inputs, perm=[1, 0, 2])
  outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
      lstm_layers,
      transposed_inputs,
      dtype='float32',
      time_major=True)
  unstacked_outputs = tf.unstack(outputs, axis=0)
  return unstacked_outputs[-1]

outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
      lstm_layers,
      transposed_inputs,
      dtype='float32',
      time_major=True)

outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
      lstm_layers,
      transposed_inputs,
      dtype='float32',
      time_major=True,
      initial_state=[tf.compat.v1.placeholder(tf.float32, shape=(200)), tf.compat.v1.placeholder(tf.float32, shape=(200))])

outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
      lstm_layers,
      transposed_inputs,
      dtype='float32',
      time_major=True,
      initial_state=[tf.compat.v1.placeholder(tf.float32, shape=(1, 200)), tf.compat.v1.placeholder(tf.float32, shape=(1, 200))])