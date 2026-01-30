import torch

a = torch.zeros((2, 2), requires_grad=True)
b = 2 * a
c = b.sum()
b.detach_()
c.backward()
print(a.grad)