import torch.nn as nn

(python)
import torch
class M(torch.nn.Module):
  def forward(self, x):
    return torch.nonzero(x)

m = M()
exported_program = torch.export.export(m, (torch.randn(64, 10),))
exported_m = exported_program.module()
print(exported_m)
exported_m.compile()
print(exported_m)
exported_m(torch.randn(64, 10))