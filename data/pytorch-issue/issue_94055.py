import torch

def fn(v0, v2):
    # v0: (3, 4, 2, 1)
    # v2: (2, 2, 1, 4, 1)
    v5 = v0.argmax(3) # v5: (3, 4, 2)
    v4 = torch.min(v2, v5) # v4: (2, 2, 3, 4, 2)
    v1 = v4.min(3).values # v1: (2, 2, 3, 2)
    return [v1]

x = torch.rand(3, 4, 2, 1).int()
y = torch.rand(2, 2, 1, 4, 1).int()
fn(x, y)
print('==== Eager mode OK! ====')

compiled = torch.compile(fn)
compiled(x, y)
print('==== torch.compile mode OK! ====')