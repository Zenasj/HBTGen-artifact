import torch
print(torch.__version__)

x = torch.randn(2, 3, 3, dtype=torch.double, device='cuda').permute(0, 2, 1)
y = torch.inverse(x)

print(y)