from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op

class ClipOpsTest(test.TestCase):

    def _testClipIndexedSlicesByNorm(self, values, indices, shape, max_norm, axes):
        values = constant_op.constant(values)
        indices = constant_op.constant(indices)
        shape = constant_op.constant(shape)
        indexed_slices = ops.IndexedSlices(values, indices, shape)
        clipped = clip_ops.clip_by_norm(indexed_slices, max_norm, axes)
        clipped = ops.convert_to_tensor(clipped)
        
    def testClipIndexedSlicesByNorm_Failed(self):
        values = [[[(- 3.0), 0.0, 0.0], [4.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 0.0, (- 1.0)]]]
        indices = [2, 6]
        # shape = [9223372036854775807, 1, 9223372036854775807]
     
        shape = [9223372036854775807, 2, 3] 
        self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, None) # crashed


    def testClipIndexedSlicesByNorm_Pass(self):
        values = [[[(- 3.0), 0.0, 0.0], [4.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 0.0, (- 1.0)]]]
        indices = [2, 6]

        shape = [10, 2, 3] 
        self._testClipIndexedSlicesByNorm(values, indices, shape, 4.0, None) # passed

if (__name__ == '__main__'):
    test.main()