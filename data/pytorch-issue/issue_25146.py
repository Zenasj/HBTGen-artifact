import torch.nn as nn

import torch
# in_channels=3, out_channels=16, stride=3
opt = torch.nn.Conv2d(3, 16, 3)
# batch_size=5, channel=3, additional_dim=1, height=224, width=224 
inputs = torch.randn(5, 3, 1, 224, 224)
opt(inputs)
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight 16 3 3 3 140728118629264, but got 5-dimensional input of size [5, 3, 1, 224, 224] instead

import torch
# in_channels=3, out_channels=16, stride=3
opt = torch.nn.Conv2d(3, 16, 3)
# batch_size=5, channel=3, additional_dim=1, height=224, width=224 
inputs = torch.randn(5, 3, 1, 224, 224)
opt(inputs)
# Expected 4-dimensional input for 4-dimensional weight 16 3 3 3, but got 5-dimensional input of size [5, 3, 1, 224, 224] instead