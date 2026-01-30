import torch.nn as nn

import torch

class TensorConstant(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, a):
    return a / torch.tensor(3)


arg = torch.arange(4).to(torch.uint16)

print(torch.export.export(TensorConstant(), (arg,)))