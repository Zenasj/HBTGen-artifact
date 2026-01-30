import torch
import torch.nn as nn

class Foo(nn.Module):
  def __init__(self):
    self.weight = nn.Parameter(torch.randn(3))
    self.tied_weight = self.weight

  def forward(self, inp):
    ...