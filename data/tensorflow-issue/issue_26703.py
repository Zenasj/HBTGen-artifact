from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class F1(tf.metrics.Metric):

  def __init__(self, **kwargs):
    super(F1, self).__init__(**kwargs)
    self.precision = tf.metrics.Precision()
    self.recall = tf.metrics.Recall()

  @property
  def updates(self):
    return self.precision.updates + self.recall.updates

  def update_state(self, y_true, y_pred):
    self.precision.update_state(y_true, y_pred)
    self.recall.update_state(y_true, y_pred)

  def result(self):
    precision = self.precision.result()
    recall = self.recall.result()
    return (2 * precision * recall) / (recall + precision)

with tf.Graph().as_default() as graph:
  precision = tf.metrics.Precision()
  precision.update_state([0, 0, 1], [1, 0, 1])
  print(precision.updates[0].graph is graph)  # True

  f1 = F1()
  f1.update_state([0, 0, 1], [1, 0, 1])
  print(f1.updates[0].graph is graph)  # False

f1 = F1()
f1.update_state([0, 0, 1], [1, 0, 1])
print(f1.updates[0].graph is graph)  # True
print(f1.updates[1].graph is graph)  # False
print(f1.updates[2].graph is graph)  # False

class F1(tf.metrics.Metric):
  def __new__(cls, *args, **kwargs):
    return tf.keras.layers.Layer.__new__(cls)