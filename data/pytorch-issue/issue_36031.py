import torch.nn as nn

import torch
from collections import namedtuple

Outputs = namedtuple('Outputs', ['val'])

class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()

    def forward(self, x, y):
        out = Outputs(val = x + y)

        # This statement doesn't execute when this model is converted to torchscript
        out.val.add_(x)
        return out

x = torch.tensor([1.0])
y = torch.tensor([3.0])

net = Foo()
print('This gives the correct output of 5:', net(x,y))
net = torch.jit.script(net)
print('But when using the torchscripted version of net you get this wrong output (gives 4):',net(x,y))

# Looking at the torchscript you can see that out.val.add_(x) has been elided.  This prints:
#  def forward(self,
#      x: Tensor,
#      y: Tensor) -> __torch__.Outputs:
#    out = __torch__.Outputs(torch.add(x, y, alpha=1),)
#    return out
print(net.code)

def forward(self, x):
        out = Outputs(val = x)
        out.val.add_(x)
        return out

@torch.jit.script
class OutputsC:
    def __init__(self, val):
        self.val = val