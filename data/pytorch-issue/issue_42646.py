import torch.nn as nn

from typing import Any, List
import torch

class TestModule(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, input: torch.Tensor) -> Any:
    ret: List[torch.Tensor] = []
    if input.shape[0] == 1:
      return input
    else:
      ret.append(input)
      return ret

m = TestModule()
m_scripted = torch.jit.script(m)