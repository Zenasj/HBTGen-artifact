import torch.nn as nn

import torch

def fn(input):
    v1 = torch.nn.functional.pad(input, pad=(1, 0))
    return torch.gt(v1, input)

x = torch.rand([1, 2, 2, 1], dtype=torch.float64) # works fine if dtype=float32
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
AssertionError: (tensor([[[False, False],
         [False, False]],

        [[False, False],
         [False, False]]]), tensor([[[False,  True],
         [False,  True]],

        [[False, False],
         [False, False]]]))
"""