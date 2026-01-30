import torch
torch.bincount(minlength=10, weights=torch.ones([1,0,1]), input=torch.tensor([10]))