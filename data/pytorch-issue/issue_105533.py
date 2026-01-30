import torch

@torch.compile()
def wh(n, a):
    a[n] = -1
    return a

z = torch.zeros(7)
for n in range(2, z.shape[0]):
    print(wh(n, z))