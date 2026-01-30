import torch

def fn(input):
    v1 = torch.dstack([input, input, input])   # works fine with [input, input]
                                               # works fine if directly passing tensor to reciprocal
    return torch.reciprocal(v1)

x = torch.tensor(True)

ret_eager = fn(x)

compiled = torch.compile(fn)
ret_compiled = compiled(x)

for r1, r2 in zip(ret_eager, ret_compiled):
    assert torch.allclose(r1, r2), (r1, r2)
print('==== Check OK! ====')

"""
Traceback (most recent call last):
  File "repro.py", line 15, in <module>
    assert torch.allclose(r1, r2), (r1, r2)
AssertionError: (tensor([[1., 1., 1.]]), tensor([[1.0000, 1.0000, 0.0115]]))
"""

print(f'==== Check OK! ==== version {torch.__version__} version_info {sys.version_info}')

import torch._inductor.config
torch._inductor.config.debug = True