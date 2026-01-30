import torch.nn as nn

import torch

def fn(input):
    rt = torch.stack([input]) # works fine if "[input, input]" is the input tensor list
    v = torch.nn.functional.silu(input, inplace=True) # any inplace operator will trigger this issue
    return rt

x = torch.tensor([1.0, 1.0])
ret_eager = fn(x.clone())

compiled = torch.compile(fn)
ret_compiled = compiled(x.clone())

for r1, r2 in zip(ret_eager, ret_compiled):
    assert torch.allclose(r1, r2), (r1, r2)
print('==== Check OK! ====')

"""
Traceback (most recent call last):
  File "repro.py", line 15, in <module>
    assert torch.allclose(r1, r2), (r1, r2)
AssertionError: (tensor([1., 1.]), tensor([0.7311, 0.7311]))
"""