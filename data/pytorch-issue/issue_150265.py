import torch.nn as nn

from torch import nn
import torch
torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
}
class a(nn.Linear):
  def __init__(self, b):
    super().__init__(128, 128)
    self.b = b
class b(nn.Parameter):
  def __new__(cls, data):
    self = torch.Tensor._make_subclass(cls, data)
    return self
A = a(b(torch.randn(12, 12)))
@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def test():
  out = 3 * A.b
  return out

test()