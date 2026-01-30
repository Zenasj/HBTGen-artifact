import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc1(x)
input_size = 2
num_classes = 2
model = TestModel(input_size, num_classes).eval()
with torch.cpu.amp.autocast():
    model = torch.jit.trace(model, torch.randn(1, input_size))

param = Parameter(...)
x0 = ...
x1 = cell(x0, param)
x2 = cell(x1, param)
...

param = Parameter(...)
x0 = ...
cached_param = param.half()
x1 = cell(x0, cached_param)
x2 = cell(x1, cached_param)
...

param = Parameter(...)
x0 = ...
x1 = cell(x0, param.half())
x2 = cell(x1, param.half())
...

import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.fc1(x)

class AMPModel(nn.Module):
    def __init__(self, module):
        super(AMPModel, self).__init__()
        self.module = module
    def forward(self, x):
        with torch.cpu.amp.autocast():
            self.module(x)

input_size = 2
num_classes = 2
model = TestModel(input_size, num_classes).eval()
wrapped_model = AMPModel(model)

model = torch.jit.trace(wrapped_model, torch.randn(1, input_size))

with torch.cpu.amp.autocast(cache_enabled=False):
    traced_model = torch.jit.trace(model, input)

with torch.cpu.amp.autocast():
    model1(input1)
    with torch.cpu.amp.autocast(cache_enabled=False):
        model2(input2)