import itertools
import torch

res = [torch.rand((5,) * d, dtype=torch.double, device='cpu') for d in range(4)]

for input in itertools.product(res, repeat=3):
    print(f"shape : {[i.shape for i in input]}")
    torch.foo(*input)