import torch
import torch.nn as nn

class M(torch.nn.Module):
  def forward(self, x):
    return x.to(x.device)

scripted = torch.jit.script(M())
print(scripted.graph)