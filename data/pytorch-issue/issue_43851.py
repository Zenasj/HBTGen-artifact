import torch
import torch.nn as nn

class MyFunc(torch.nn.Module):
  def __init__(self):
    super(MyFunc, self).__init__()

  def forward(self, input, w, b):
    return torch.nn.functional.linear(input, w, b)

input = torch.rand([2, 2048])
w = torch.rand([128, 2048])
b = torch.rand([128])

m = MyFunc()
ms = torch.jit.script(m)

r1 = m(input, w, b)
r2 = ms(input, w, b)
r3 = m(input, w, b)

print(torch.equal(r1, r2))
print(torch.equal(r1, r3))