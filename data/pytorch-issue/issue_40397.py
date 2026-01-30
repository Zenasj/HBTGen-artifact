import torch
x = torch.randn(3, 3)
y = torch._softmax(x, -1, False)
print(y)

def _softmax(self: Tensor, dim: _int, half_to_float: _bool) -> Tensor: ...

# Appease the type checker: it can't deal with direct setting of globals().
# Note that we will see "too many" functions when reexporting this way; there
# is not a good way to fix this problem.  Perhaps, try to redesign VariableFunctions
# so that this import is good enough
if False:
    from torch._C._VariableFunctions import *