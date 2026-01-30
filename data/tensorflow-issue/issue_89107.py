import tensorflow as tf
import numpy as np
import unittest

class TestConvertToTensor(unittest.TestCase):
    def test_convert_float(self):
        # Test conversion of a float to a tensor
        float_data = 3.14
        tensor = tf.convert_to_tensor(float_data)
        self.assertTrue(isinstance(tensor, tf.Tensor))
        self.assertEqual(tensor.dtype, tf.float32)
        self.assertEqual(tensor.numpy(), float_data)
if __name__ == '__main__':
    unittest.main()