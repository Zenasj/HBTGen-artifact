a = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, 4)
out = a * 2
out.sum().backward()
# Calling swap_tensors here would fail due to the reference held by AccumulateGrad node, which is not cleaned up after backward
# torch.utils.swap_tensors(a, b)
del out
# Calling swap_tensors here would pass
torch.utils.swap_tensors(a, b)

a = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, 4)
out = a * 2
out.sum().backward()
# Calling swap_tensors here is ok
torch.utils.swap_tensors(a, b)
# If we ever backward to the AccumulateGrad node it will error that it was poisoned by swap_tensors

import torch
import torch.nn as nn
m = nn.Linear(3, 5)
inp = torch.randn(2, 3)
out = m(inp)
out.sum().backward()
m.cpu()