import torch

a = torch.tensor(1, dtype=torch.uint32)
b = torch.tensor(2, dtype=torch.uint32)
torch.bitwise_xor(a, b)