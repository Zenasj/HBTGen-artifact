import torch

def fn(input):
    v = input.select(2, 0)
    return v.add_(1) # works w/ non-inplace op "add"

x = torch.rand([1, 2, 1, 2, 1, 2]) # works fine w/ other shapes like [1, 1, 1, 1, 1, 1]

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')