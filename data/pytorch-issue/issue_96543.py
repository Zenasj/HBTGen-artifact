import torch

x = torch.tensor([0.0 + 0.0j, -1.0568e-23 + 0.0j], requires_grad=True)
y = x.abs()
y.backward(torch.ones_like(y))
print("gradient:", x.grad)