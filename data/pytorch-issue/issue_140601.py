import torch

x = torch.ones(2, 2, dtype=torch.float64)
w = torch.ones(2, 2, dtype=torch.float64)
s = torch.tensor(2.2) 

z = x.lerp_(w, s)