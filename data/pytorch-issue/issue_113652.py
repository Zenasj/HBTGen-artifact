import torch.nn as nn

import torch


class SubModelImpl(torch.autograd.Function):
  @staticmethod
  def symbolic(g, x):
    return g.op('custom::Identity', x).setType(x.type())

  @staticmethod
  def forward(ctx, x):
    return x


class SubModel(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return SubModelImpl.apply(x)


class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.submodel_trace = torch.jit.trace(SubModel(), torch.randn((5, ), dtype=torch.float32))

  def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
    for _ in range(n):
      x = self.submodel_trace(x)
    return x


model_script = torch.jit.script(Model())
x = torch.randn((5, ), dtype=torch.float32)
n = 3

torch.onnx.export(
  model_script,
  (x, n),
  '/tmp/test.onnx',
  verbose=True)