import torch

def cool_name(x):
    return x.sin()

def fn(x):
    return torch.no_grad(cool_name)

x = torch.ones([])
result = fn(x)
print(result.__name__)
result = torch.compile(fn, backend="eager", fullgraph=True)(x)
print(result.__name__)