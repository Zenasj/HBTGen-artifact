import torch
import unittest
from common_utils import TestCase

class TestMkldnn(TestCase):
    def test_mul(self):
        size = 1024
        x = torch.randn(size, dtype=torch.float32)
        mx = x.to_mkldnn()
        self.assertEqual((x * x), (mx * mx).to_dense())

if __name__ == '__main__':
    unittest.main()