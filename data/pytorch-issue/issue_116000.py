import torch.nn as nn

import torch

a = torch.rand(1, 1, 10, 10)
b = torch.nn.Conv2d(1, 1, 1, 1)
b.cuda()(a.cuda())