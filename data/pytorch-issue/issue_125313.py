import torch.nn as nn

import torch
import torch.nn.functional as F

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.linear1(x)
        torch._dynamo.graph_break()
        return F.relu(x) * self.scale

x = torch.ones(1, 10)
mod = BasicModule()
mod = torch.compile(mod)
print(mod(x))

mod(x).sum().backward()

compile_fx_inner

convert_frame