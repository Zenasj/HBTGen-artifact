import numpy as np
import random

import tensorflow as tf
from tensorflow.python.framework import test_util


class WeirdTest(tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_test(self):
    with self.cached_session():
      x = tf.random.normal([])
      self.assertAllClose(x, x)


if __name__ == '__main__':
  tf.random.set_seed(42)
  tf.test.main()

# Here is how the (shared) graph is built
model = build_model()

# Here is how operations in the graph are evaluated
# including what kind of inputs are feeding and what kind of outputs are fetched
do_something(model)

x = tf.random.normal([])

with tf.Session() as sess:
    first = sess.run(x)
    second = sess.run(x)
    np.testing.assert_allclose(first, second)

input = tf.placeholder(foo)
noise = tf.random.normal([])
noisy_output = input + noise

# do something

with tf.Session() as sess:
    for i in range(10):
        y = sess.run(noisy_output, feed_dict=bar)