import torch

a = torch.randn(3, 4).t()
b = torch.randn(3, 4).t()
out = torch.empty(4, 3)
print(out.stride())  # prints 3,1 as expected
torch.add(a, b, out=out)
print(out.stride())  # prints 1,4
out = torch.empty(4, 3)  # reset out strides
torch.sin(a, out=out)
print(out.stride())  # prints 1,4