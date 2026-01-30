import torch.nn as nn

import torch
import torch.nn
import torch.nn.functional

x = torch.ones(1,3,2,2)
print('default ceil_mode', torch.nn.functional.avg_pool2d(x, 4).shape)
print('ceil_mode=True', torch.nn.functional.avg_pool2d(x, 4, stride=4, padding=0, ceil_mode=True, count_include_pad=True).shape)
print('ceil_mode=False', torch.nn.functional.avg_pool2d(x, 4, stride=4, padding=0, ceil_mode=False, count_include_pad=True).shape)