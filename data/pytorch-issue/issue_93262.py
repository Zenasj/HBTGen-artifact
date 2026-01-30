import torch.nn as nn

import torch

def fn(input):
    return torch.nn.functional.interpolate(input, size=[1, 1], mode="bilinear")
    # works fine if `mode` is set to other values such as "nearest" and "bicubic"

x = torch.rand([2, 8, 7, 10])
ret_eager = fn(x)

compiled = torch.compile(fn)
ret_compiled = compiled(x)

for r1, r2 in zip(ret_eager, ret_compiled):
    assert torch.allclose(r1, r2), (r1, r2)
print('==== Check OK! ====')

"""
Traceback (most recent call last):
  File "repro.py", line 14, in <module>
    assert torch.allclose(r1, r2), (r1, r2)
AssertionError: (
    tensor([[[0.5000]], [[0.5813]], [[0.5536]], [[0.2098]], [[0.3442]], [[0.7828]], [[0.6382]], [[0.3412]]]), 
    tensor([[[0.1909]], [[0.8426]], [[0.0925]], [[0.1988]], [[0.0879]], [[0.0518]], [[0.7094]], [[0.6926]]]))
"""