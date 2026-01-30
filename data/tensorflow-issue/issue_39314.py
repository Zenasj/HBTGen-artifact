from tensorflow import keras
from tensorflow.keras import layers

class Count(tf.keras.metrics.Metric):
  def __init__(self, name=None, dtype=None, **kwargs):
    super(Count, self).__init__(name, dtype, **kwargs)
    self.count = tf.Variable(0)

  def update_state(self, y_true, y_pred, sample_weight=None):
    first_tensor = tf.nest.flatten(y_true)[0]
    batch_size = tf.shape(first_tensor)[0]
    self.count.assign_add(batch_size)

  def result(self):
    return tf.identity(self.count)

import tensorflow as tf

class Count(tf.keras.metrics.Metric):
  def __init__(self, name=None, dtype=None, **kwargs):
    super(Count, self).__init__(name, dtype, **kwargs)
    self.count = tf.Variable(0)

  def update_state(self, y_true, y_pred, sample_weight=None):
    first_tensor = tf.nest.flatten(y_true)[0]
    batch_size = tf.shape(first_tensor)[0]
    self.count.assign_add(batch_size)

  def result(self):
    return tf.identity(self.count)


class PrintInfo(tf.keras.callbacks.Callback):
  def on_train_batch_end(self, batch, logs):
    print('Batch number: {}'.format(batch))
    print('Samples seen this epoch: {}'.format(logs['counter']))

model = tf.keras.Sequential([tf.keras.layers.Dense(1)])
model.compile(optimizer='sgd', loss='mse', metrics=[Count(name='counter')])
x, y = tf.ones((10, 10)), tf.ones((10, 1))
model.fit(x, y, batch_size=2, callbacks=[PrintInfo()], verbose=2)