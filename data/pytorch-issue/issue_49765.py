import torch.nn as nn

import torch

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.submod = torch.nn.Linear(3, 4)
    self.submod = None

  def forward(self, inputs):
    return inputs

m = TestModule()
tm = torch.jit.trace(m, torch.tensor(1.))