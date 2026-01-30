import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU

# Tests included in `TestMyTensorOperators` should assert the actual
# logic of the operators assuming they are run on a real device.
#
# As of today practically all our device-specific test cases behave
# similar to this class.
class TestMyTensorOperators(TestCase):
    def test_my_operator(self, device):
        ...

    @onlyCPU
    def test_my_other_operator(self, device):
        ...

# Instantiates tests of `TestMyTensorOperators` to be run on real
# devices (i.e. CPU, CUDA), note that this function does NOT instantiates
# tests for the meta backend anymore.
instantiate_device_type_tests(TestMyTensorOperators, globals())

# Since the semantics of a meta tensor is very different than a regular
# tensor, any operator logic that is specific to the meta backend
# should be tested in a separate `TestCase`.
class TestMyTensorOperatorsOnMeta(TestCase):
    ...