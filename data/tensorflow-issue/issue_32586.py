import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class CellWrapper(tf.keras.layers.AbstractRNNCell):

    def __init__(self, cell):
        super(CellWrapper, self).__init__()
        self.cell = cell

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.cell.get_initial_state(
            inputs=inputs, batch_size=batch_size, dtype=dtype)

    def call(self, inputs, states, training=None, **kwargs):
        assert training is not None


cell = tf.keras.layers.LSTMCell(32)
cell = CellWrapper(cell)
cell = tf.keras.layers.StackedRNNCells([cell])

rnn = tf.keras.layers.RNN(cell)
inputs = tf.random.uniform([4, 7, 16])
rnn(inputs, training=True)