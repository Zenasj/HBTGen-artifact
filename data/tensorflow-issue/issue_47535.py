import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow

class GRU(tf.Module):
  def __init__(self):
    super(GRU, self).__init__()
    self.cell = tf.keras.layers.GRUCell(units=2, dropout=0.1, recurrent_dropout=0.1)

  @tf.function
  def bad_infer(self, inputs, states):
    o, h = self.cell(inputs=inputs, states=states, training=True)
    return o, h

  def good_infer(self, inputs, states):
    o, h = self.cell(inputs=inputs, states=states, training=True)
    return o, h

gru = GRU()
inputs = tf.ones((1, 2))
states = gru.cell.get_initial_state(inputs)
targets = tf.ones(1, 2)

gru.good_infer(inputs=inputs, states=states)
gru.variables

gru.bad_infer(inputs=inputs, states=states)
gru.variables