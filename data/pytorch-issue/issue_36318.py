import torch

x = torch.tensor(1.0, dtype=torch.half)

x + 1.0
x - 1.0
x * 1.0
x / 1.0

x ** 2.0
torch.sqrt(x)
torch.max(x)
torch.ceil(x)