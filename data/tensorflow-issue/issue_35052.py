import random
from tensorflow.keras import optimizers

import os
import sys
import timeit

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

# tf.compat.v1.disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(8)
os.environ['OMP_NUM_THREADS'] = '1'

bucket = int(1e7)

class MyModel(keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()

  def build(self, input_shape):
    self.user_emb = self.add_weight(
        shape=(bucket + 1, 32),
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(),
        name="user_emb")
    self.item_emb = self.add_weight(
        shape=(bucket + 1, 32),
        dtype=tf.float32,
        initializer=tf.keras.initializers.TruncatedNormal(),
        name="item_emb")
    self.bias = tf.Variable(0.0)

  def call(self, inputs):
    user_id, item_id = inputs
    user_id = tf.reshape(user_id, [-1])
    item_id = tf.reshape(item_id, [-1])
    out = tf.gather(self.user_emb, user_id) * tf.gather(self.item_emb, item_id)
    out = tf.reduce_sum(out, axis=1, keepdims=True) + self.bias
    out = tf.sigmoid(out)
    return out


def main():

  def py_func(feats):
    label = feats['labels']
    return (feats['user_id'], feats['item_id']), label

  model = MyModel()

  dataset = tf.data.Dataset.from_tensor_slices({
      "user_id": np.random.randint(bucket, size=[1000, 1]),
      "item_id": np.random.randint(bucket, size=[1000, 1]),
      "labels": np.random.randint(2, size=[1000, 1])
  }).map(py_func)

  model.compile(
      keras.optimizers.SGD(0.01), 'binary_crossentropy', metrics=['AUC'])

  # model.run_eagerly = True
  model.fit(
      dataset,
      shuffle=False,
      workers=1,
      epochs=1000)

if __name__ == '__main__':
  main()