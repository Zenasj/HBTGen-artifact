import random

import torch

types = [torch.uint8, torch.int8, torch.short, torch.int, torch.long, torch.half, torch.float, torch.double]

print('Testing linspace:')
for type in types:
    print(type, torch.linspace(-2, 2, 10, dtype=type))

import torch, random

types = [torch.int8, torch.short, torch.int, torch.long, torch.float, torch.double]
a = -120
b = 123
for p in range(1, 8):
    n = 5**p
    print('Testing with n={0}:'.format(n))
    for dtype in types:
        _from = random.random()
        to = _from + random.random()
        res1 = torch.linspace(_from, to, 137, dtype=dtype)
        res2 = torch.tensor((), dtype=dtype)
        torch.linspace(_from, to, 137, dtype=dtype, out=res2)
        print(dtype, (res1-res2).abs().max(), sep='\t')
    
        expected_lin = torch.tensor([a + (b-a)/(n-1)*i for i in range(n)], dtype=torch.double).to(dtype)
        actual_lin = torch.linspace(a, b, n, dtype=dtype)
        print(dtype, (actual_lin-expected_lin).abs().max(), sep='\t')