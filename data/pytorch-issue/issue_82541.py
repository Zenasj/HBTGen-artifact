import torch

device = "mps"
a = torch.tensor([[]], device=device, requires_grad=True)
b = torch.tensor([[]], device=device, requires_grad=True)
y = torch.true_divide(a, b)
y.sum().backward()  # seg fault