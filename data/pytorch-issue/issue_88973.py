import torch
device = torch.device("mps")
v = torch.arange(10,device=device)
v = v[v>5]