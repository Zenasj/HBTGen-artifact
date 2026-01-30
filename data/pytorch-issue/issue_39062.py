import torch.nn as nn

import torch, torch.nn as nn

class CustomLayer(nn.Module):
  def __init__(self):
    super().__init__()
    self.threshold = nn.Parameter(torch.randn(1)).to('cuda')
  def forward(self, input):
    return input

model = nn.Sequential(CustomLayer()).to('cuda')
list(model.parameters())

[]

to('cuda')