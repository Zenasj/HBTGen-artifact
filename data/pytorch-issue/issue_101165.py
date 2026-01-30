import torch
import torch.nn as nn

mod = nn.Sequential(*[nn.Linear(2,2)])


@torch.compile
def seq():
  inp = torch.randn(2,2)

  for m in mod:
      inp = m(inp)

seq()