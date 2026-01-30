# File: test_foo.py

import torch
import numpy as np

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import OpInfo, DecorateInfo, toleranceOverride, tol, all_types_and, _dispatch_dtypes

op = OpInfo(
    'foo',
    op=(lambda x:x),
    dtypesIfCPU=_dispatch_dtypes((torch.bfloat16,)),
    dtypesIfCUDA=_dispatch_dtypes(),
    decorators=[DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=1e-03, rtol=1e-03)}), 'TestFoo', 'test_foo')])


class TestFoo(TestCase):

    @ops([op])
    def test_foo_success(self, device, dtype, op):
        self.assertEqual(torch.tensor(21.3750, dtype=torch.bfloat16), np.array(21.390625, dtype=np.float32), exact_dtype=False)

    @ops([op])
    def test_foo_failing(self, device, dtype, op):
        self.assertEqual(torch.tensor(21.3750, dtype=torch.bfloat16), np.float32(21.390625), exact_dtype=False)

instantiate_device_type_tests(TestFoo, globals())

if __name__ == '__main__':
    run_tests()