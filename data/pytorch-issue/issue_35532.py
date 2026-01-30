import torch

a = torch.tensor([0 + 0j, 1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.float)
b = torch.exp(a)