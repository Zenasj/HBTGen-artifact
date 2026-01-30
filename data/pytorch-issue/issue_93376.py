import torch

def fn(input):
    return torch.rsub(input, input, alpha=2)

x = torch.rand([1])

ret_eager = fn(x)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x)
print('==== torchcomp mode OK! ====')