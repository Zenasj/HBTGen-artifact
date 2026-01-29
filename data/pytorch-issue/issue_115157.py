# torch.rand(1, 45, 1, 1, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m0 = nn.MaxPool2d(kernel_size=(1, 1), stride=1, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        m1 = self.m0(x)
        abs_1 = torch.abs(m1)
        pad = torch.nn.functional.pad(abs_1, (0, 24, 34, 0), 'replicate')
        add = torch.add(pad, pad)
        return (pad, add)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 45, 1, 1, dtype=torch.float32)

