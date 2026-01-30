import torch

def func(a, b):
    return a + b

def func1(**kwargs):
    return func(a=1, **kwargs)

c_f = torch.compile(func1, fullgraph=True)
c_f(b=torch.rand([2]))