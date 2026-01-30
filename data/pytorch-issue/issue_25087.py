import torch
torch.histc(torch.tensor([]).cuda(), bins=4, min=0, max=1)