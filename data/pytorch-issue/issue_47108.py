import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(5))

    def forward(self, x):
        return torch.dot(self.W, x)

mod = Test()
print(fx.symbolic_trace(Test())(5))