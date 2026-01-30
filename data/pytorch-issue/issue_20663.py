import torch

x = torch.tensor([1.], requires_grad=True)
y = x.clone()
z = x * 2
print(torch.autograd.grad(z, x))  # 2, OK
print(torch.autograd.grad(z, y))  # RuntimeError

import torch

a = torch.tensor([1., 3., 5.], requires_grad=True)
b = torch.tensor([2., 4., 6.], requires_grad=True)
a = a.unsqueeze(1)
b = b.unsqueeze(1)
z = torch.cat((a, b), dim=1)
a1 = z[:, 0]
b1 = z[:, 1]
c = torch.matmul(a1.t(), b1)

print(torch.autograd.grad(c, a, retain_graph=True)[0]) #returns b
print(torch.autograd.grad(c, b, retain_graph=True)[0]) #returns a
print(torch.autograd.grad(c, z)[0])