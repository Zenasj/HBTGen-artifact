import torch

def fn(i0):
    o = i0.unsqueeze_(0) # AssertionError: While executing %unsqueeze_
    # o = i0.unsqueeze(0) # works fine
    return o

x = torch.rand(1)
print(fn(x.clone())) # works fine
print('==== Eager mode OK! ====')

compiled = torch.compile(fn, dynamic=False, fullgraph=False)
print(compiled(x.clone())) # AssertionError: While executing %unsqueeze_
print('==== torchcomp OK! ====')