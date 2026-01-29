# torch.rand(4, dtype=torch.bfloat16, requires_grad=True, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu6 = nn.ReLU6()  # Core module under test

    def forward(self, x):
        return self.relu6(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, dtype=torch.bfloat16, requires_grad=True, device="cuda")

