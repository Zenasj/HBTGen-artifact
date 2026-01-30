import torch

a = torch.tensor([1,2,3])
val = torch.tensor(2.5, requires_grad=True)

ind = torch.searchsorted(a, val)