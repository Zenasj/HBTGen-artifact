import torch.nn as nn

import torch
torch.manual_seed(0)
data = torch.randn([512, 3, 32, 32])   # cifar-shaped batch
module = torch.nn.Conv2d(3, 4, kernel_size=3, padding=0)
module(data)