import torch.nn as nn

import torch
import torch._dynamo

class A(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.sum()

class C(A):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return A.forward(self, input)


gm, _ = torch._dynamo.export(C(), torch.randn(4, 5))
print(gm)