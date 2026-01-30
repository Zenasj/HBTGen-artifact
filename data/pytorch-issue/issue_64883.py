import torch.nn as nn

import torch
import warnings

torch.use_deterministic_algorithms(True, warn_only=False)
print('warn_only=False')

try:
    torch.arange(10).cuda().bincount()
except RuntimeError as e:
    print('bincount throws an error, as expected')
    print()
else:
    assert False

torch.use_deterministic_algorithms(True, warn_only=True)
print('warn_only=True')

with warnings.catch_warnings(record=True) as w:
    torch.arange(10).cuda().bincount()
    print(f"We can catch the warning for bincount: {w}")
    print()

module = torch.nn.AvgPool3d(3)
input = torch.randn(2, 3, 3, 3, requires_grad=True).cuda()
res = module(input)
grad = torch.ones_like(res)

torch.use_deterministic_algorithms(True, warn_only=False)
print('warn_only=False')

try:
    res.backward(grad)
except RuntimeError as e:
    print('avg_pool3d_backward throws an error, as expected')
    print()
else:
    assert False

torch.use_deterministic_algorithms(True, warn_only=True)
print('warn_only=True')

with warnings.catch_warnings(record=True) as w:
    res.backward(grad)
    print(f"We cannot catch the warning for avg_pool3d_backward: {w}")
    print()