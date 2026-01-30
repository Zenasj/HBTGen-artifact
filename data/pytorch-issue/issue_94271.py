import torch.nn as nn

import torch_directml
import torch
from torch import nn

gpu_device = torch_directml.device()
  
class Model(nn.Module):
     def __init__(self):
          super().__init__()
          self.net = nn.InstanceNorm2d(3, affine=True)

     def forward(self, x):
          return self.net(x)

model = Model().to(gpu_device)
x = torch.randn(10, 3, 64, 64).to(gpu_device)
y = model(x)