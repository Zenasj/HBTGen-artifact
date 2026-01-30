import torch.nn as nn

import torch
import torch._dynamo
from torch import Tensor

class MyInnerModule(torch.nn.Module):
  def __init__(self):
    pass
     
  def forward(self, a:Tensor, b:Tensor = torch.ones((2, 3))):
    c = a + b
    d = c + a
    e = c + b
    f = d + e
    return f

class MyModule(torch.nn.Module):
  def __init__(self):
    super(MyModule, self).__init__()
    self.m = MyInnerModule()

  def forward(self, a:Tensor):
    res = self.m(a)
    return res
 
my_module = MyModule()
my_module = torch._dynamo.optimize("aot_ts")(my_module)

inp = torch.ones((2, 3))
res = my_module(inp)