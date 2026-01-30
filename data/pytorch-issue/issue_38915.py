import torch.nn as nn

import torch

def time_bn(bn, x):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    x = bn(x)
    end.record()
    torch.cuda.synchronize()
    # print('++++++++++ CUDA Time +++++++++++')
    # print(start.elapsed_time(end))
    return x

shape = (50, 50, 50, 50)
a = torch.rand(shape).cuda()
b = a.flatten(0, -2) + 1

# Commented resp during nvprof.
bna = torch.nn.BatchNorm2d(50).cuda()
time_bn(bna, a)
bnb = torch.nn.BatchNorm1d(50).cuda()
time_bn(bnb, b)