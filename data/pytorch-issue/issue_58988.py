import torch
import numpy as np

from torch.testing._internal.common_utils import TestCase


class TestFoo(TestCase):
    def test_bar(self):
        x = torch.ones((1,), dtype=torch.float16)
        y = np.ones((1,), dtype=np.float32)

        self.assertEqual(x, y, exact_dtype=True)

assert isinstance(next(iter(np.array((1.0,)))), numbers.Number)