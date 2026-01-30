import torch

import unittest
import unittest.mock

from torch.testing._internal.common_distributed import skip_if_not_multigpu


patcher = unittest.mock.patch("torch.cuda.device_count", return_value=1)
patcher.start()


class TestCase(unittest.TestCase):
    @skip_if_not_multigpu
    def test_foo(self):
        assert False