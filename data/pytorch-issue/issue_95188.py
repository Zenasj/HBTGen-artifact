import torch.nn as nn

import torch
from copy import deepcopy

def test_linear(x_shape):
    x = torch.randn(x_shape)
    linear_cpu = torch.nn.Linear(320,320)
    linear_mps = deepcopy(linear_cpu).to('mps')
    cpu_result = linear_cpu(x)
    mps_result = linear_mps(x.to('mps'))
    torch.testing.assert_close(mps_result, cpu_result, check_device=False, msg="MPS result not equal to CPU result for shape " + str(x_shape))

# 512x512 image size
test_linear([2, 4096, 320])
# 1152x1152 image size
test_linear([2, 20736, 320])