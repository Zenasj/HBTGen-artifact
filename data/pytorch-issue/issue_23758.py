import torch

a = torch.tensor([1], dtype=torch.uint8)
b = torch.tensor([1.0]) # whatever, does not matter
c = a & ( b < 2.0)