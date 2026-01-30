import torch.nn as nn

import torch
from torch.fx.experimental.proxy_tensor import make_fx

class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=2, end_dim=3)

x = torch.randn(3, 5, 4, 5)
m = make_fx(MyModule(), tracing_mode="symbolic")(x)

for node in m.graph.nodes:
    if isinstance(node.target, torch._ops.OpOverloadPacket):
        print(type(node.target))  # <class 'torch._ops.OpOverloadPacket'>
        print(node.target.overloads())  # ['default', 'int']