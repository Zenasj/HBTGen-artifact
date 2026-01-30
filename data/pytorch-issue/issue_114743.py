import torch.nn as nn

import torch
import torch._functorch.aot_autograd

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))
        self.register_parameter("weight1", weight)
        self.weight2 = self.weight1

    def forward(self, x):
        t = torch.matmul(self.weight1, x) + torch.matmul(self.weight2, x)
        return t.sum()

def nop(*args):
    pass

model = Model()
aot_model = torch._functorch.aot_autograd.aot_module(model, fw_compiler=nop, bw_compiler=nop)
aot_model(torch.rand(10))