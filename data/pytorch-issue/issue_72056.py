import torch.nn as nn

import torch
input = torch.FloatTensor([[ 0.1735,  0.2191],
        [-0.1377,  0.3467],
        [-0.2162,  0.2021],
        [-0.1177,  0.2157],
        [-0.1232,  0.2620],
        [-0.1578,  0.4038],
        [-0.0209,  0.3298],
        [-0.0639,  0.2317],
        [-0.3914,  0.2106],
        [-0.4211,  0.3420]])
target =torch.LongTensor([ 0, -1,  0,  0,  0,  0,  0,  0, -1,  0])
weight = torch.FloatTensor([1,16])
reduction = 'mean'
ignore_index = -1
label_smoothing = .1
torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)