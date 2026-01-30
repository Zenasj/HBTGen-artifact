import torch
x = torch.randn(10, 10, device='cuda')
y = x.inverse()
print(y)