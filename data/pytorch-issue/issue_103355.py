import os
os.environ['PYTORCH_TEST_WITH_DYNAMO']='1'

import torch
from torch import tensor
from torch.testing._internal.common_utils import TestCase
import torch._dynamo
import unittest
import warnings

class TestFoo(TestCase):
    def test_boolean_shape_mismatch(self, device='cpu'):
        arr = torch.ones((5, 4, 3), device=device)

        index = tensor([True], device=device)
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        index = torch.ByteTensor(4, 4).to(device).zero_()
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

    def test_bool_indices(self, device='cpu'):
        v = torch.randn(5, 7, 3, device=device)
        boolIndices = torch.tensor([True, False, True, True, False], dtype=torch.bool, device=device)
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        boolIndices = torch.tensor([True, False, False], dtype=torch.bool, device=device)
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        with warnings.catch_warnings(record=True) as w:
            self.assertEqual(v[boolIndices].shape, v[uint8Indices].shape)
            self.assertEqual(v[boolIndices], v[uint8Indices])
            self.assertEqual(v[boolIndices], tensor([True], dtype=torch.bool, device=device))
            print("\n\nw=",[f"{x.filename}:{x.lineno} {x.message}" for x in w],"\n\n")
            self.assertEqual(len(w), 2)

if __name__ == "__main__":
    unittest.defaultTestLoader.sortTestMethodsUsing = lambda *args: -1
    unittest.main()