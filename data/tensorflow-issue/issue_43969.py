import tensorflow as tf
from tensorflow.python.framework.test_util import assert_no_garbage_created

class TestStandardTrainer(tf.test.TestCase):
  @assert_no_garbage_created
  def test_dataset_map(self):
      data_tensor = {'x': tf.constant([1, 2, 3], dtype=tf.float32),
                     'y': tf.constant([4, 5, 6], dtype=tf.float32)}
      dataset = tf.data.Dataset.from_tensor_slices(data_tensor)

      # We split the data in 2
      dataset = dataset.map(lambda x: (x, x))

import unittest
unittest.main(argv=['first-arg-is-ignored'], exit=False)