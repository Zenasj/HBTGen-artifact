import torch.nn as nn

import torch
import torch._dynamo 

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(2, 3))
    
    def forward(self, x):
        return self.p + x

with torch.device("meta"):
    m = MyModule()

torch._dynamo.export(m, torch.ones(2, 3))