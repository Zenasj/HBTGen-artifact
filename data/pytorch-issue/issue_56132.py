import torch

t = torch.tensor([1,2,3])
torch.logsumexp(t, 0)