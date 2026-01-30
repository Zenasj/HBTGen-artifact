import torch
torch.manual_seed(0)
a = torch.zeros(5).long()
a.bernoulli_()