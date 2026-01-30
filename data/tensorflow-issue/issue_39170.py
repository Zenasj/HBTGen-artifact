from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test

class ShuffleAndRepeatFusionTest(test_base.DatasetTestBase):

  def testShuffleAndRepeatFusion(self):
    dataset = dataset_ops.Dataset.range(10)
    get_next = self.getNext(dataset)

if __name__ == "__main__":
  test.main()

import tensorflow as tf
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import test
tf.compat.v1.app.flags.DEFINE_string('f', '', 'kernel')
class ShuffleAndReplaceFusion(test_base.DatasetTestBase):
  def testShuffleAndRepeatFusion(self):
    dataset = dataset_ops.Dataset.range(10)
    get_next = self.getNext(dataset)

if __name__ == "__main__":
  test.main()