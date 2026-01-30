import torch

x = torch.ones(2, device='mps:0')
x.logsumexp(dim=-1, keepdim=True)