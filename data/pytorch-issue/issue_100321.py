import torch
x = torch.randn(7, 11, 13)
dim = 0
index = torch.tensor([0, 4, 2])
values = torch.randn(3, 13)
ans = x.index_add(dim, index, values.requires_grad_()) # I appended the requires gradient as the error wouldn't be thrown without it, even though it should as described in the documentation