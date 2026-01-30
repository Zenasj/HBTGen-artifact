tup = (1, 2)
self.a = tup[0]
self.b = tup[1]

import torch
import torch.nn as nn

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = 1
        self.b = 2
    def forward(self, inputs):
        self.a, self.b = (1, 2)
        return inputs

a = torch.jit.script(A())