from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import tensorflow as tf
from tensorflow_core.python.keras.models import Model, Sequential
from tensorflow_core.python.keras.layers.core import Dense, Activation, Lambda, Reshape
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.recurrent import RNN, StackedRNNCells
from tensorflow_core.lite.experimental.examples.lstm.rnn_cell import TFLiteLSTMCell, TfLiteRNNCell
from tensorflow_core.lite.experimental.examples.lstm.rnn import dynamic_rnn
from tensorflow_core.python.ops.rnn_cell_impl import LSTMStateTuple
from tensorflow_core.lite.python.interpreter import Interpreter
from tensorflow_core.python.ops.rnn_cell_impl import MultiRNNCell

def get_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)


def buildMultiCell(cells):
    return MultiRNNCell(cells)


def buildRNNLayer(inputs, rnn_cells, initial_state=None):
  """Build the lstm layer.

  Args:
    inputs: The input data.
    num_layers: How many LSTM layers do we want.
    num_units: The unmber of hidden units in the LSTM cell.
  """
  # Assume the input is sized as [batch, time, input_size], then we're going
  # to transpose to be time-majored.
  transposed_inputs = tf.transpose(inputs, perm=[1, 0, 2])
  outputs, new_state = dynamic_rnn(
      rnn_cells,
      transposed_inputs,
      initial_state=initial_state,
      dtype='float32',
      time_major=True)
  unstacked_outputs = tf.unstack(outputs, axis=0)
  # update_op = get_state_update_op(initial_state, new_state)
  return unstacked_outputs[-1], new_state


def build_rnn_lite(model, state=False):
    tf.reset_default_graph()
    # Construct RNN
    cells = []
    for layer in range(3):
        if model == 'LSTMLite':
            cells.append(TFLiteLSTMCell(192, name='lstm{}'.format(layer)))
        else:
            cells.append(TfLiteRNNCell(192, name='rnn{}'.format(layer)))

    rnn_cells = Lambda(buildMultiCell, name='multicell')(cells)
    states = get_state_variables(1, rnn_cells)
    if state:
        spec_input = Input(shape=(5, 64,), name='rnn_in', batch_size=1)
        x, new_states = Lambda(buildRNNLayer, arguments={'rnn_cells': rnn_cells, 'initial_state': states}, name=model.lower())(spec_input)
        updated_states = Lambda(get_state_update_op, arguments={'new_states': new_states}, name='update_state')(states)
    else:
        spec_input = Input(shape=(5, 64,), name='rnn_in')
        x, new_states = Lambda(buildRNNLayer, arguments={'rnn_cells': rnn_cells}, name=model.lower())(spec_input)
        updated_states = Lambda(get_state_update_op, arguments={'new_states': states}, name='update_state')(states)

    out = Dense(64, activation='sigmoid', name='fin_dense')(x)
    return Model(inputs=spec_input, outputs=[out, updated_states])

model = build_rnn_lite('LSTMLite', True)

###### TF LITE CONVERSION
sess = tf.keras.backend.get_session()
input_tensor = sess.graph.get_tensor_by_name('rnn_in:0')
output_tensor = []
output_tensor.append(sess.graph.get_tensor_by_name('fin_dense/Sigmoid:0'))
all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
# output_tensor = sess.graph.get_tensor_by_name('fin_dense/Sigmoid:0')

# imp_tensor = []
for ten in all_tensors:
    if 'update_state/Assign' in ten.name:
        output_tensor.append(ten)

converter = tf.lite.TFLiteConverter.from_session(sess, [input_tensor], output_tensor)
# Note: It will NOT work without enabling the experimental converter!
# `experimental_new_converter` flag.
converter.experimental_new_converter = True
tflite_seg = converter.convert()