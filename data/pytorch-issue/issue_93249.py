import torch

def fn(x):
    o = x.min(1).values
    return o

x = torch.rand((2, 8, 2), dtype=torch.float16) # AssertionError
# x = torch.rand((2, 7, 2), dtype=torch.float16) # works fine
fn(x)
print('==== CPU Eager mode OK! ====')

compiled = torch.compile(fn)
compiled(x)
print('==== CPU compiled mode OK! ====')

compiled(x.cuda())
print('==== GPU compiled mode OK! ====')