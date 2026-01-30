import torch
import torch_backend  # Your 3rd-party device's extension

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


class TestBinaryUfuncs(TestCase):
    def test_abc(self,):
        pass


instantiate_device_type_tests(TestBinaryUfuncs, globals())

if __name__ == "__main__":
    run_tests()