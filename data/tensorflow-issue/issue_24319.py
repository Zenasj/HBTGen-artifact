import random
from tensorflow import keras
from tensorflow.keras import layers

def reset_states(self, states=None):
    import datetime
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    batch_size = self.input_spec[0].shape[0]
    if not batch_size:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the batch size by passing a '
                       '`batch_shape` argument to your Input layer.')
    # initialize state if None
    if self.states[0] is None:
      if _is_multiple_state(self.cell.state_size):
        self.states = [
            K.zeros([batch_size] + tensor_shape.as_shape(dim).as_list())
            for dim in self.cell.state_size
        ]
      else:
        self.states = [
            K.zeros([batch_size] +
                    tensor_shape.as_shape(self.cell.state_size).as_list())
        ]
    elif states is None:
      if _is_multiple_state(self.cell.state_size):
        now = datetime.datetime.now()
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state,
                      np.zeros([batch_size] +
                               tensor_shape.as_shape(dim).as_list()))
        print(f"LSTM reset time: {datetime.datetime.now() - now}")  # TODO
      else:
        K.set_value(self.states[0], np.zeros(
            [batch_size] +
            tensor_shape.as_shape(self.cell.state_size).as_list()))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, '
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if _is_multiple_state(self.cell.state_size):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != tuple([batch_size] +
                                tensor_shape.as_shape(dim).as_list()):
          raise ValueError(
              'State ' + str(index) + ' is incompatible with layer ' +
              self.name + ': expected shape=' + str(
                  (batch_size, dim)) + ', found shape=' + str(value.shape))
        # TODO(fchollet): consider batch calls to `set_value`.
        K.set_value(state, value)

import tensorflow as tf
import numpy as np

COUNT_LSTMS = 200

BATCH_SIZE = 100
UNITS_INPUT_OUTPUT = 5
UNITS_LSTMS = 20
BATCHES_TO_GENERATE = 2
SEQUENCE_LENGTH = 20

# build model
my_input = tf.keras.layers.Input(batch_shape=(BATCH_SIZE,
                                              None,
                                              UNITS_INPUT_OUTPUT))
my_lstm_layers = [tf.keras.layers.LSTM(units=UNITS_LSTMS,
                                       stateful=True,
                                       return_sequences=True)(my_input)
                  for _ in range(COUNT_LSTMS)]
my_output_layer = tf.keras.layers.Dense(UNITS_INPUT_OUTPUT)
my_output = tf.keras.layers.TimeDistributed(my_output_layer)(
    tf.keras.layers.concatenate(my_lstm_layers, axis=-1))
my_model = tf.keras.Model(my_input, my_output)


# generation
pred_input = np.random.randn(BATCH_SIZE, 1, UNITS_INPUT_OUTPUT)

for batch in range(BATCHES_TO_GENERATE):
    print('resetting states')
    my_model.reset_states()
    print(f"start generation of batch {batch}")
    for _ in range(SEQUENCE_LENGTH):
        pred_input = my_model.predict(pred_input, batch_size=BATCH_SIZE)