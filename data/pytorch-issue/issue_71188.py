import torch.nn as nn

import torch

torch.manual_seed(0)
x = torch.rand((1, 3, 10, 10))
y = torch.nn.functional.interpolate(x, (1,1), align_corners=False, mode='bilinear')
print(y)

import torch

torch.manual_seed(0)
x = torch.rand((1, 3, 10, 10))
y = torch.nn.functional.interpolate(x, (1,1), align_corners=True, mode='bilinear')
print(y)