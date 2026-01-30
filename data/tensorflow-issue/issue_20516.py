import tensorflow as tf
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.training.checkpointable import util

class Model(checkpointable.Checkpointable):

  def __init__(self):
    self.cell = tf.nn.rnn_cell.BasicLSTMCell(4)
    self.cell = tf.nn.rnn_cell.BasicLSTMCell(4)
    # self.cell = tf.nn.rnn_cell.BasicLSTMCell(4)
    out = self.cell(tf.constant([[1.]]), self.cell.zero_state(1, tf.float32))
    self.optimizer = tf.train.AdamOptimizer()
    self.optimizer.minimize(tf.reduce_sum(out[0]))
    self.session = tf.Session()
    self.checkpoint = tf.train.Checkpoint(model=self)

  def init(self):
    print('Init')
    self.session.run(tf.global_variables_initializer())

  def save(self):
    print('Save')
    self.checkpoint.save('./tmp/', self.session)

  def restore(self):
    print('Restore')
    latest = tf.train.latest_checkpoint('./tmp/')
    load_status = self.checkpoint.restore(latest)
    print(util._serialize_object_graph(self.checkpoint, None))
    load_status.assert_consumed().run_restore_ops(self.session)

  def print(self):
    print(self.session.run(self.cell._kernel))


m = Model()
m.init()
m.print()
m.save()
m.restore()
m.print()