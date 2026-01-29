# torch.rand(1, 3, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize weights as a parameter to ensure groups is fixed at construction
        self.weight = nn.Parameter(torch.rand(3, 1, 3, 3, dtype=torch.float32))
        self.groups = int(self.weight.shape[0])  # groups=3 here

    def forward(self, x):
        return F.conv2d(x, self.weight, groups=self.groups)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

