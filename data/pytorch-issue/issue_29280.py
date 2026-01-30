import torch
import timeit
import numpy
device="cpu"
torch.set_num_threads(1)

def fn(input, dim):
    return input.sum(dim=dim)

sizes = [[500,500,4], [128,128,128]]
for size in sizes:
    input_ = torch.randn(*size, device=device)
    for dim in range(3):
        print('Size ', size, 'dim ', dim, 'time:', timeit.timeit('fn(input_, dim)', number=100, globals=globals()))