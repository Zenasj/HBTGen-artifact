import torch

# Create input tensors with requires_grad=True
a = torch.empty((2, 3), requires_grad=True)
b = torch.empty((3, 4), requires_grad=True)
c = torch.empty((2, 4))

# Should throw RuntimeError: "functions with out=... arguments don't support automatic differentiation"
torch.tensordot(a, b, dims=([1], [0]), out=c)