py
import torch
from torch.testing._internal.common_utils import TestCase

assertEqual = TestCase().assertEqual

actual = torch.sparse_csr_tensor([0, 2, 4], [0, 1, 0, 1], [[1, 11], [2, 12] ,[3, 13] ,[4, 14]])
expected = torch.stack([actual[0].to_dense(), actual[1].to_dense()])
assertEqual(actual, expected)