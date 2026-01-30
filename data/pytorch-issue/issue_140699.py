import torch 
v = torch.full((0,0), 2.77406, dtype=torch.double)
g = torch.full((0,), 1.11933, dtype=torch.double)
dim = 0
torch._weight_norm(v, g, dim)