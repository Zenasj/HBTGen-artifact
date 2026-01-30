import torch.nn as nn

import torch
from torch import nn
torch.manual_seed(100)

class Temp(nn.Module):
    def __init__(self):
        super(Temp, self).__init__()

    def forward(self, input, pad, mode, value):
        return torch.nn.functional.pad(input, pad, mode, value)


model = Temp()
cmodel = torch.compile(model)

input = torch.randn(1,1,1)
print(f"The input is: ", input)
input = input.to('cuda')
pad = [10, 10]
mode = 'constant'
value = 0
print(f"Eager result: ", model(input, pad, mode, value))
print(f"Compiled result: ", cmodel(input, pad, mode, value))