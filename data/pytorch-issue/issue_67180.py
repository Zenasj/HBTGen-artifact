import torch
a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
c = a * b
c1 = torch.nansum(c)  # or torch.nanmean

bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
print(bgrad1)