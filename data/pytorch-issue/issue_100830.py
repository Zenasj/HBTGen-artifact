py
import torch

torch.manual_seed(420)

x = torch.randn(1, 10, dtype=torch.bfloat16)
y = torch.randn(1, 10, dtype=torch.bfloat16)

class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.linear = nn.Linear(10, 10, bias=True).to(dtype=torch.bfloat16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out1 = self.linear(input1)
        out2 = self.relu(input2)
        out = out1 + out2
        return out

func = ExampleModel()

with torch.no_grad():
    func.train(False)
    res1 = func(x, y) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x, y)
    print(res2)
    # IOT instruction (core dumped)

import torch
import torch.nn as nn
torch.manual_seed(420)
from torch._inductor import config

config.cpp.weight_prepack=False

x = torch.randn(1, 18, dtype=torch.bfloat16)
y = torch.randn(1, 18, dtype=torch.bfloat16)

class ExampleModel(nn.Module):

    def __init__(self):
        super(ExampleModel, self).__init__()
        self.linear = nn.Linear(18, 18, bias=True).to(dtype=torch.bfloat16)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out1 = self.linear(input1)
        out2 = self.relu(input2)
        out = out1 + out2
        return out

func = ExampleModel()

with torch.no_grad():
    func.train(False)
    res1 = func(x, y) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x, y)
    print(res2)