import torch.nn as nn

import torch

class Norm(torch.nn.Module):
  def forward(self,x):
    y = x.new_empty(1, 80, 5)
    torch.nn.init.normal_(y)
    return y

n = Norm()

torch.onnx.export(n, (torch.rand(5, 10)), "n.onnx", opset_version=13, verbose=True)