from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
import numpy as np
from tensorflow.python.eager import def_function


class XlaShardingTest(test_util.TensorFlowTestCase):  
  def test_dot_split(self):
    @def_function.function
    def split_helper(tensor):
      device_mesh = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
      split_tensor = xla_sharding.mesh_split(tensor, device_mesh, [0, 1])
      self.assertIsInstance(split_tensor, ops.Tensor)
      split_sharding = xla_sharding.get_tensor_sharding(split_tensor)
      split_shape = xla_sharding.get_sharding_tile_shape(split_sharding)
      expected_shape = [2, 4]
      self.assertEqual(expected_shape, split_shape)

      y_tensor = array_ops.ones([8, 8], dtype=dtypes.float32)
      y_split = xla_sharding.mesh_split(y_tensor, device_mesh, [0, 1])
      result = math_ops.matmul(split_tensor, y_split)
      device_mesh = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
      result = xla_sharding.mesh_split(result, device_mesh, [0, 1])
      result = math_ops.sqrt(result)
      result = xla_sharding.mesh_split(result, device_mesh, [1, 0])
      return result

    in_tensor = 2 * np.sqrt(2) * array_ops.ones([8, 8], dtype=dtypes.float32)
    result = split_helper(
        array_ops.ones([8, 8], dtype=dtypes.float32))
    self.assertAllEqual(in_tensor, result)


if __name__ == "__main__":
    xlasharding = XlaShardingTest()
    xlasharding.test_dot_split()