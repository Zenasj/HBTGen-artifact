import torch
from torch.autograd import Variable

x = Variable(torch.randn(5, 5, dtype=torch.double, device='cuda'), requires_grad=True)
y = x.to_sparse().to_dense()
torch.autograd.grad(y, x, torch.randn(5, 5, dtype=torch.double, device='cuda', requires_grad=True), create_graph=True)