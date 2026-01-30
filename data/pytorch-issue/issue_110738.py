import torch
compiled = torch.compile(lambda x: (x.shape[0] > 1) + (x.shape[0] < 3), dynamic=True)
compiled(torch.zeros([2]))

import torch
def f(x):
    x += 1
    y = x.shape[0]
    z = y / x.shape[1]
    y -= z
    x += 1
    return y, x

compiled = torch.compile(f, backend="inductor", dynamic=True)
res = compiled(torch.zeros((2, 3)))
print(res)