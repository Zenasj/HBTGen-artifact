import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
# ERROR
        super().__init__()

    def forward(self, x):
        return x

input = torch.rand(1, 2, 3)
model = Model()

print("# Test 1: directly executing model: " , end="")
model(input)
print("passed!")

print("# Test 2: jit model: ", end="")
torch.jit.script(model)
print("passed!")

def __init__(self):
# ERROR
        super().__init__()

def __init__(self):
# NO ERROR ANYMORE
    super().__init__()