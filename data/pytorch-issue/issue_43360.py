import torch

x = torch.randn([], dtype=torch.float32)
y = torch.randn([], dtype=torch.float64)
print(torch.result_type(x, y))
print(torch.add(x, y).dtype)
print(torch.atan2(x, y).dtype)

torch.float64
torch.float64
torch.float32