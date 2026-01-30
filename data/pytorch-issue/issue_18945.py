import torch
import torch.nn.functional as F

x = torch.rand(10, 4, requires_grad=True)
y = torch.rand((10, 4))
loss = F.binary_cross_entropy(x, y)
loss2 = torch.autograd.grad(loss, x, create_graph=True)

loss2[0].sum().backward()

def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()