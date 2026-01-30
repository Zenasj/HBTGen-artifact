import torch

import itertools

@torch.compile(backend='eager', fullgraph=True, dynamic=True)
def f(x):
    r = itertools.accumulate([x, x])
    return x * 2

f(torch.randn(2, 3)) 
# torch._dynamo.exc.Unsupported: call_function BuiltinVariable(sum) [TensorVariable(), TensorVariable()] {}