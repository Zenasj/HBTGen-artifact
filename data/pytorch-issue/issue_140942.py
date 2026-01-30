import torch

grad_output = torch.full((2,8,2,2,2,10,9,9,), 9.87654e+09, dtype=torch.float)
self = torch.full((2,10,9,4,8,0,3,), 0, dtype=torch.float)
padding = [-1, -1]

torch.ops.aten.reflection_pad1d_backward(grad_output, self, padding)