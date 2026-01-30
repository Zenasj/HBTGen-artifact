import torch

c = torch.empty(0, dtype=torch.bfloat16)
torch.add(torch.ones(2), 1, out=c)