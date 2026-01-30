import torch
x = torch.zeros(1)
x.requires_grad_(True)
y = x ** 2

dy = torch.autograd.grad(y, x, create_graph=True)[0]
z = 2 * x

d2y = torch.autograd.grad(dy, x, create_graph=True)[0]
dz = torch.autograd.grad(z, x, create_graph=True)[0]

print(d2y.requires_grad)
print(dz.requires_grad)

d3y = torch.autograd.grad(d2y, x, create_graph=True)[0]
# d2z = torch.autograd.grad(dz, x, create_graph=True)[0]  # Throws an error

print(d3y.requires_grad)

True
False
False

True
True
True