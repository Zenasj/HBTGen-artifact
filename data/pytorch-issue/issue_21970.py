import torch
from torch.utils.checkpoint import checkpoint


def f(x, w):
    return torch.einsum('ab,ab->ab', [x, w])


def g(x, w):
    return torch.einsum('ab,ab->a', [x, w])


def h(x, w):
    return torch.einsum('ab,ab->a', [x, w]).clone()


# Function f, works
x = torch.ones(1, 1)
w = torch.ones(1, 1).requires_grad_()
y = checkpoint(f, x, w)
z = y.sum()
z.backward()
print(y, z, w.grad)

# Function g, fails
x = torch.ones(1, 1)
w = torch.ones(1, 1).requires_grad_()
y = checkpoint(g, x, w)
z = y.sum()
z.backward()
print(y, z, w.grad)

# Function h, works
x = torch.ones(1, 1)
w = torch.ones(1, 1).requires_grad_()
y = checkpoint(h, x, w)
z = y.sum()
z.backward()
print(y, z, w.grad)