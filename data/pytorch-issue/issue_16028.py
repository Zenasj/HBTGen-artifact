import torch

a = torch.ones([2, 300],requires_grad=True).float()
p = torch.softmax(a, dim=1).log()
x = p.sum()
x.backward()
a.grad.sum() # = 0

a = torch.ones([2, 300],requires_grad=True).float()
p = torch.log_softmax(a, dim=1)
x = p.sum()
x.backward()
a.grad.sum() # = tensor(7.1526e-05)