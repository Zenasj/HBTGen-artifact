import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np


class RNNGenerator(keras.layers.Layer):
    def __init__(self, rnn_dim, rnn_layer_num, seq_len, seq_width, batch_size, **kwargs):
        self.rnn_dim = rnn_dim              # lstm hidden dim
        self.rnn_layer_num = rnn_layer_num  # number of lstm cell layers
        self.seq_len = seq_len              # length of squence
        self.seq_width = seq_width          # width of squence
        self.batch_size = batch_size

        self._cells = {}

        super().__init__(**kwargs)

    def build(self, input_shape):
        # input dense
        self._dense_in = keras.layers.Dense(self.rnn_dim)

        # many lstm cell layers
        for i in range(self.rnn_layer_num):
            cell = keras.layers.LSTMCell(units=self.rnn_dim)
            self._cells[i] = cell

        # output dense
        self._dens_out = keras.layers.Dense(self.seq_width, activation="sigmoid")

        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.squeeze(inputs, 1)

        # init cells' states
        states = getattr(self, 'states', None)
        if states is None:
            states = {}
            for i, cell in self._cells.items():
                init_cell_states = [tf.random.uniform([self.batch_size, self.rnn_dim]),
                                    tf.random.uniform([self.batch_size, self.rnn_dim])]
                states[i] = init_cell_states

        # --- prev outputs as current inputs
        rand_prev = tf.random.uniform([self.batch_size, self.seq_width], dtype=tf.float32)
        prev_inputs = tf.concat([rand_prev, inputs], -1)
        outputs = []
        for _ in range(self.seq_len):
            cell_inputs = self._dense_in(prev_inputs)
            for i, cell in self._cells.items():
                cell_outputs, states[i] = cell(cell_inputs, states[i])
                cell_inputs = cell_outputs
            step_outputs = self._dens_out(cell_outputs)
            prev_inputs = tf.concat([step_outputs, inputs], -1)

            outputs.append(tf.expand_dims(step_outputs, 1))

        # reserve cells' state in current batch
        self.states = states

        return tf.concat(outputs, 1)

# @tf.function
def main():
    batch_size = 17
    seq_width = 4
    rand_input_dim = 3
    inputs = keras.layers.Input(shape=[1, rand_input_dim])
    outputs = RNNGenerator(rnn_dim=64, rnn_layer_num=2, seq_len=10, seq_width=seq_width, batch_size=batch_size)(inputs)
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(1)(outputs)
    outputs = tf.nn.sigmoid(outputs)
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # model.summary()

    X = np.random.rand(batch_size, 1, rand_input_dim).astype(np.float32)
    y = np.zeros([batch_size, 1])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
    model.fit(X, y, batch_size=32, epochs=10)


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow import keras
import numpy as np


class RNNGenerator(keras.layers.Layer):
  def __init__(self, rnn_dim, rnn_layer_num, seq_len, seq_width, batch_size, **kwargs):
    self.rnn_dim = rnn_dim              # lstm hidden dim
    self.rnn_layer_num = rnn_layer_num  # number of lstm cell layers
    self.seq_len = seq_len              # length of squence
    self.seq_width = seq_width          # width of squence
    self.batch_size = batch_size

    self._cells = {}

    super().__init__(**kwargs)

  def build(self, input_shape):
    # input dense
    self._dense_in = keras.layers.Dense(self.rnn_dim)

    # many lstm cell layers
    for i in range(self.rnn_layer_num):
      cell = keras.layers.LSTMCell(units=self.rnn_dim)
      self._cells[i] = cell

    # output dense
    self._dens_out = keras.layers.Dense(self.seq_width, activation="sigmoid")

    # init cells' states
    states = getattr(self, 'states', None)
    if states is None:
      states = {}
      for i, cell in self._cells.items():
        init_cell_states = [tf.random.uniform([self.batch_size, self.rnn_dim]),
                            tf.random.uniform([self.batch_size, self.rnn_dim])]
        states[i] = init_cell_states
      self.states = states

    super().build(input_shape)

  # @tf.function
  def call(self, inputs):
    inputs = tf.squeeze(inputs, 1)

    states = self.states.copy()

    # --- prev outputs as current inputs
    rand_prev = tf.random.uniform([self.batch_size, self.seq_width], dtype=tf.float32)
    prev_inputs = tf.concat([rand_prev, inputs], -1)
    outputs = []
    for _ in range(self.seq_len):
      cell_inputs = self._dense_in(prev_inputs)
      for i, cell in self._cells.items():
        cell_outputs, states[i] = cell(cell_inputs, states[i])
        cell_inputs = cell_outputs
      step_outputs = self._dens_out(cell_outputs)
      prev_inputs = tf.concat([step_outputs, inputs], -1)

      outputs.append(tf.expand_dims(step_outputs, 1))

    # reserve cells' state in current batch
    for state_, state in zip(tf.nest.flatten(self.states), tf.nest.flatten(states)):
      state_ = state

    return tf.concat(outputs, 1)

# @tf.function
def main():
  batch_size = 17
  seq_width = 4
  rand_input_dim = 3
  inputs = keras.layers.Input(shape=[1, rand_input_dim])
  outputs = RNNGenerator(rnn_dim=64, rnn_layer_num=2, seq_len=10, seq_width=seq_width, batch_size=batch_size)(inputs)
  outputs = keras.layers.Flatten()(outputs)
  outputs = keras.layers.Dense(1)(outputs)
  outputs = tf.nn.sigmoid(outputs)
  model = keras.models.Model(inputs=inputs, outputs=outputs)

  # model.summary()

  X = np.random.rand(batch_size, 1, rand_input_dim).astype(np.float32)
  y = np.zeros([batch_size, 1])

  model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
  model.fit(X, y, batch_size=32, epochs=10)


if __name__ == "__main__":
  main()