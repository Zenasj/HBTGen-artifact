import torch
x = torch.randn(5, 5)
y = torch.randn(5, 5)
result = torch.trapezoid(y, x=x, dx=0.01)