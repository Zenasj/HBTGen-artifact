import torch.nn as nn

import torch
import torch._dynamo

class M(torch.nn.Module):
    def forward(self, x):
        return x + torch.nn.Parameter(torch.ones(3, 3))

inps = (torch.ones(3, 3),)
torch._dynamo.export(M())(*inps)