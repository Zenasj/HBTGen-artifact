import torch

def fn(x):
    return (x * 0.5j).sum()


x = torch.ones(1, dtype=torch.double, requires_grad=True)
o = fn(x)
o.backward() # Shouldn't work
print(x.grad)