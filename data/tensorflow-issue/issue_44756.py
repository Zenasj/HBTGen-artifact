from tensorflow import keras

import tensorflow as tf
from tensorflow.python.framework.test_util import assert_no_garbage_created

class TestMetric(tf.test.TestCase):
  @assert_no_garbage_created
  def test_metric(self):
    metric = tf.keras.metrics.Mean('test', dtype=tf.float32)

import unittest
unittest.main(argv=['first-arg-is-ignored'], exit=False)