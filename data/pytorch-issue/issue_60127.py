import torch

leaf = torch.tensor(..., requires_grad=True)
tmp = leaf * 2
loss = tmp.sum()
torch.autograd.grad(loss, inputs=(tmp, leaf))