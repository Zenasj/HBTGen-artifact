import torch

t = torch.rand(2, 3).to(torch.float16)
torch.round(t, decimals=3)