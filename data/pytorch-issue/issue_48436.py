import torch

torch.manual_seed(42)

shapes = [
    (4, 4),
    (16, 16),
    (3, 5, 17),
    (128, 128),
    (512, 512),
    (5, 127, 451),
    (128, 8, 8),
    (128, 16, 16),
    (242, 7, 3)
]

def get_diff(x, y):
    a, b = torch.testing._compare_tensors_internal(
                x.double().abs(), y.double().abs(),
                rtol=0, atol=0, equal_nan=False)
    # return (a, b)
    idx_was = b.find('was')
    idx_space = b.find(' ', idx_was + 5)

    return float(b[idx_was + 4: idx_space])

for shape in shapes:
    x = torch.randn(*shape, dtype=torch.float, device='cuda')
    xe = x.double()
    
    y = torch.svd(x)
    ye = torch.svd(xe)

    a = get_diff(y[0], ye[0])
    print(str(shape).ljust(25), f'{a : .3e}')

import torch
import os

torch.manual_seed(42)

shapes = [
    (4, 4),
    (16, 16),
    (3, 5, 17),
    (128, 128),
    (512, 512),
    (5, 127, 451),
    (128, 8, 8),
    (128, 16, 16),
    (242, 7, 3)
]


if not os.path.isdir('data'):
    os.mkdir('data')

for shape, i in zip(shapes, range(len(shapes))):
    x = torch.randn(*shape, dtype=torch.double, device='cuda')
    u, s, v = torch.svd(x)

    torch.save(s, f'data/{i}.pt')

import torch

shapes = [
    (4, 4),
    (16, 16),
    (3, 5, 17),
    (128, 128),
    (512, 512),
    (5, 127, 451),
    (128, 8, 8),
    (128, 16, 16),
    (242, 7, 3)
]

print('| shape | max_diff |')
print('| --- | --- |')

for shape, i in zip(shapes, range(len(shapes))):
    s_cusolver = torch.load(f'data-cusolver/{i}.pt')
    s_magma = torch.load(f'data-magma/{i}.pt')

    max_diff = (s_cusolver - s_magma).abs().max()

    print('|', str(shape), '|', max_diff, '|')