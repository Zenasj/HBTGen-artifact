import torch.nn as nn

import torch
import os
from torch import nn
import time


def perf_test(postfix):
    inp = torch.load(os.path.join(base_dir, 'input_{}.pth'.format(postfix)))
    out = torch.load(os.path.join(base_dir, 'output_{}.pth'.format(postfix)))
    conv = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, padding=1)
    conv.load_state_dict(torch.load(os.path.join(base_dir, 'conv2d_{}.pth'.format(postfix))))
    start = time.time()
    res = conv(inp)
    end = time.time()
    assert torch.all(res == out).item() == 1
    print(postfix, 'time:', end - start)


base_dir = 'results/conv2d_performance'
with torch.no_grad():  # does not matter either way
    for cntr1 in range(3):
        for cntr2 in range(3):
            perf_test('slow')
        for cntr2 in range(3):
            perf_test('fast')

python
def clear_from_denormalized_values(model):
    eps = pow(2, -125)
    for p in model.parameters():
        p.detach()[torch.abs(p) < eps] = 0

...
clear_from_denormalized_values(conv)
...