import torch

def fn(x, y):
    o = x.masked_fill(y, 0) # torch.Tensor.masked_fill-1 # v4: (4, 1040, 12)
    return o

x = torch.rand(2, 4, 8)
y = torch.ones(2, 4, 1).bool()

"""Other shapes may eliminate the error
x = torch.rand(2, 4, 6)
y = torch.ones(2, 4, 1).bool()
"""

eager = fn(x, y)
compiled = torch.compile(fn)
comp = compiled(x, y)

assert torch.allclose(
    eager, comp,
), '\n'.join([
    '',
    f'>>> eager',
    str(eager),
    f'>>> comp',
    str(comp),
])
print(f'==== finished! ====')