import torch.nn as nn

import torch
pool = torch.nn.AvgPool2d(4, stride=4, padding=2, ceil_mode=True)
x = torch.ones(1, 1, 9, 9)
print(pool(x).shape)

import torch
pool = torch.nn.AvgPool2d(4, stride=4, padding=2, ceil_mode=True)
x = torch.ones(1, 1, 9, 9)
torch.onnx.export(pool, x, 'm.onnx')