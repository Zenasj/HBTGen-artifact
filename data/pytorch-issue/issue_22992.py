import torch

a = torch.tensor([1,2,3])
b = torch.tensor([True, False, True])
c = torch.tensor([1,2,3])
torch.masked_select(a, b, out=c)