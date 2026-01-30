import torch

global_a = torch.ones(...)

def foo():
    global_a.add_(1)

# this won't remove the inplace call, even though we're "functionalizing"
functionalize(foo)()