import torch.nn as nn

import torch
from torch.testing._core import _compare_tensors_internal

x = torch.randn(2, 3, 6, 6, device='cuda', dtype=torch.float)\
        .to(memory_format=torch.channels_last)

x_cpu = x.cpu()

out = torch.nn.functional.adaptive_max_pool2d(x, (2, 2))
out_cpu = torch.nn.functional.adaptive_max_pool2d(x_cpu, (2, 2))

_a, _b = _compare_tensors_internal(out.cpu(), out_cpu, rtol=1e-5, atol=1e-5, equal_nan=False)
if _a:
    print('good')
else:
    raise RuntimeError(_b)