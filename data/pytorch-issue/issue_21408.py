import torch

def f(x, y):
    d = y.sum()
    return x * d

x = torch.randn(3,3, requires_grad=True)
y = torch.randn(3,3, requires_grad=True, dtype=torch.float64)
tf = torch.jit.trace(f, (x, y))
z = tf(x, y)
z.backward(torch.randn_like(x))

def f(x, y):
    d = y.sum()
    return x * d

x = torch.randn(3,3, requires_grad=True)
y = torch.randn(3,3, requires_grad=True, dtype=torch.float64)
z = f(x, y)
z.backward(torch.randn_like(x))