import torch.nn as nn

import torch
class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.a = 11

    def forward(self, x):
        return self.a

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.sub = SubModule()

    def forward(self, x):
        self.sub.a = 1
        return self.sub.a * 20

m = TestModule()
input = torch.randn(2, 2)
sm = torch.jit.script(m)
output_s = sm(input)