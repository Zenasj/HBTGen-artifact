import torch
import torch.nn as nn


class TestModule(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

  def forward(self, x, flag: bool):
    y = []
    if flag:
      ml = self.mod_list
    else:
      ml = self.mod_list
    for m in ml:
      y.append(m(x))
    y = torch.stack(y)
    return y.sum()


s = torch.jit.script(TestModule())

ml = self.mod_list
if flag:
  ml = self.different_mod_list
for m in ml:
  y.append(m(inputs))

if flag:
  for m in self.different_mod_list:
    y.append(m(inputs))
else:
  for m in self.mod_list:
    y.append(m(inputs))