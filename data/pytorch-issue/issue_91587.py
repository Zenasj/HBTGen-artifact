# torch.rand(B, C, L, dtype=torch.float64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters based on the second example in the issue (float64, groups=0 fixed to 1)
        self.weight = nn.Parameter(torch.rand(3, 3, 3, dtype=torch.float64))
        self.bias = nn.Parameter(torch.rand(3, dtype=torch.float64))
        self.stride = [2]
        self.padding = 2
        self.output_padding = [1]
        self.groups = 1  # Fixed from original groups=0 to avoid error
        self.dilation = [1]  # Default dilation from second example

    def forward(self, x):
        return F.conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 4, dtype=torch.float64)

