import torch
import torch.nn as nn

def double(x):
    return x * 2

class Double(nn.Module):
    def __init__(self):
        super(Double, self).__init__()
        self.dble = double

    def forward(self, input):
        return self.dble(input)

class Add(nn.Module):
    def __init__(self):
        super(Double, self).__init__()
        self.add = torch.add

    def forward(self, input):
        return self.add(input)