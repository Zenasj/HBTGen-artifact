import random

import torch
from torch.autograd import profiler
torch.set_num_threads(1)

def dense_layer(input, w, b):
    return torch.addmm(b, input, w)

if __name__ == '__main__':
    torch.random.manual_seed(1234)
    a = torch.randn(100000, 10)
    b = torch.randn(10, 10)
    c = torch.randn(10)

    with profiler.profile() as prof:
        for i in range(1000):
            dense_layer(a, b, c)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=3))

    traced = torch.jit.trace(dense_layer, (a, b, c))
    # traced = torch.jit.script(dense_layer)
    with profiler.profile() as prof2:
        for i in range(1000):
            traced(a, b, c)
    print(prof2.key_averages().table(sort_by='cpu_time_total', row_limit=3))