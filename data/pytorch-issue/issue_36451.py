import torch

a = torch.tensor((1.,), dtype=torch.half)
b = torch.tensor((1.,), dtype=torch.half)
torch.isclose(a, b)