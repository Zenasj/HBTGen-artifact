import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

class M(nn.Module):
  def forward(self, x):
    return F.unfold(x, kernel_size=(2, 3))

input = torch.randn(2, 5, 3, 4)
torch.onnx.export(M(), input, "/dev/null", verbose=True, opset_version=11)