import tensorflow as tf


class Cell(tf.nn.rnn_cell.RNNCell):
  
  def __init__(self, state_size, reuse=None):
    super(Cell, self).__init__(_reuse=reuse)
    self.__state_size = state_size
    self.__encoder = tf.make_template("encoder", self.encoder)
  
  @property
  def state_size(self):
    return self.__state_size
  
  @property
  def output_size(self):
    return self.state_size

  def zero_state(self, batch_size, dtype):
    return tf.zeros([batch_size, self.state_size])

  def encoder(self, prev_state, obs):
    inputs = tf.concat([prev_state, obs], -1)
    return tf.layers.dense(inputs, self.state_size, None)

  def call(self, inputs, prev_state):
    state = self.__encoder(prev_state, inputs)
    return state, state


inputs = tf.placeholder(tf.float32, [32, 2, 20])

rnn_cell = Cell(20)
outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)

self.__encoder.updates = []