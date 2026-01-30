import torch

def foo(x):
    try:
         return torch.foo(x) # throws runtime error
    except RuntimeError:
         return x