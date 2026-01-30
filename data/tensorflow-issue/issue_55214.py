from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test

class TensorsTest(test.TestCase):

    def test_is_tensor_array(self):
        # this case will pass if using [1] in element_shape
        return list_ops.empty_tensor_list(element_shape=constant_op.constant([[1]], dtype=dtypes.int32),
                                          element_dtype=dtypes.int32) 

if (__name__ == '__main__'):
    test.main()