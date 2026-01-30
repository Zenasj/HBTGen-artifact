import torch.nn as nn

import torch

import pickle

with open('failing_input.pkl', 'rb') as f:
    data = pickle.load(f)

args = data['batched_arg']
kwargs = data['kwarg_values']  # groups = 4

i = args[0]  # torch.Size([2, 4, 6, 6])
w = args[1]  # torch.Size([4, 1, 3, 3])
b = args[2][:, 0]  # torch.Size([4, 2]) (bias is batched)

o = torch.nn.functional.conv2d(i, w, b, **kwargs)

print(o[0])

import torch

i = torch.rand(2, 4, 6, 6)
w = torch.rand(4, 1, 3, 3)
b = torch.rand(8)[::2]

o = torch.nn.functional.conv2d(i, w, b, groups=4)

i = i.to(torch.float64)
w = w.to(torch.float64)
b = b.to(torch.float64)

o2 = torch.nn.functional.conv2d(i, w, b, groups=4)

print((o - o2).abs().max())