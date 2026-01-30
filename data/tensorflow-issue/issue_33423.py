import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CompatV1LSTM(tf.keras.layers.Layer):
  def __init__(self, n_hidden = 8):
    super(CompatV1LSTM, self).__init__()
    self.lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    print('Layer Init')

  def build(self, input_shape):
    self.lstm_cell.build(input_shape)


  def call(self, x):
    x = tf.dtypes.cast(x, tf.dtypes.float32, name='Converted_floats')
    _X = tf.unstack(x, time_steps, 1)

    print('Shape of x', x.shape)
    print('Shape of input after unstack',len(_X))
    print('Shape of first element', _X[0].shape)

    output, state = tf.compat.v1.nn.static_rnn(cell=self.lstm_cell,
                                               inputs=_X, dtype=tf.float32)

    return tf.matmul(output[-1], weights['out']) + biases['out']