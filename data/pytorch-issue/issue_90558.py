import torch.nn as nn

import torch
import torch.nn.functional as F
import torch._dynamo

torch._dynamo.config.print_graph_breaks = True

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale

class SuperModule(BasicModule):
    def forward(self, x):
        x = super().forward(x)           # <--------- Look at this line
        return x + 10.0

model = SuperModule().eval()
opt_model = torch._dynamo.optimize("inductor")(model)
x = torch.rand([10, 10])
print(opt_model(x))

import torch
import torch.nn.functional as F
import torch._dynamo

torch._dynamo.config.print_graph_breaks = True

class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale

class SuperModule(BasicModule):
    def forward(self, x):
        x = BasicModule.forward(self, x)     # <--------- Look at this line
        return x + 10.0

model = SuperModule().eval()
opt_model = torch._dynamo.optimize("inductor")(model)
x = torch.rand([10, 10])
print(opt_model(x))