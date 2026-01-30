import torch.nn as nn

import torch

class Conv2dCell(torch.nn.Module):
  def __init__(self):
    super(Conv2dCell, self).__init__()

  def forward(self, x):
    conv = torch.nn.Conv2d(1, 3, 3, stride=1)
    output = conv(x)
    return output

m = Conv2dCell()
scripted_m = torch.jit.script(m)