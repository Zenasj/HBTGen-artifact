# in test_torch.py
import torch

mps = torch.device("mps")
a = torch.ones(2, 3, 4, device=mps)
print(a * 2)