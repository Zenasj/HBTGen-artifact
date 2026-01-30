import torch

def fn(x, y):
    t = torch.linalg.vector_norm(x)
    return torch.remainder(y, t)

x = torch.rand([1], dtype=torch.float16)
y = torch.rand([1], dtype=torch.float16)

ret_eager = fn(x, y)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
print('==== torchcomp compilation OK! ====')

ret_compiled = compiled(x, y)
print('==== torchcomp mode OK! ====')