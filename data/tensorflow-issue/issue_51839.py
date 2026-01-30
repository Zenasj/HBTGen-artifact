import tensorflow as tf
import numpy as np
import os

def test(xyz_batch,  k):
    indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    dist = np.zeros((1000, 1000, 20), dtype=np.float32)
    return indices, dist


while True:
    ret = tf.py_function(test, [0,0], [tf.int64, tf.int64])

def test(indices,  dist):
    # indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    # dist = np.zeros((1000, 1000, 20), dtype=np.int64)
    return indices, dist


while True:
    indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    dist = np.zeros((1000, 1000, 20), dtype=np.int64)
    ret = tf.py_function(test, [indices,dist], [tf.int64, tf.int64])

def test(indices,  dist):
    indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    dist = np.zeros((1000, 1000, 20), dtype=np.int64)
    return indices, dist

while True:
    indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    dist = np.zeros((1000, 1000, 20), dtype=np.int64)
    test(indices, dist)

def test(xyz_batch,  k):
    indices = np.zeros((1000, 1000, 20), dtype=np.int64)
    dist = np.zeros((1000, 1000, 20), dtype=np.float32)
    # print(indices.type)
    # print(xyz_batch.type)
    return indices, dist


while True:
    indices, dist = tf.numpy_function(test, [0,0], [tf.int64, tf.int64])

class _DefaultGraphStack(stack.DefaultStack):  # pylint: disable=protected-access
  """A thread-local stack of objects for providing an implicit default graph."""